from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
from sentence_transformers import InputExample
from torch.utils.data import Dataset

from utils import process_retrieval_ducoment

try:
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - optional dependency for graph GNNs
    Data = None

from models.qwen3 import Qwen3Retriever

_TOOL_VOCAB_CACHE_TAG = "train_only_v1"
_GRAPH_DATASET_CACHE_TAG = "train_only_v2"


@dataclass
class ExtraToolNode:
    """仅在大图之外出现的工具节点，测试/验证时视为孤立节点。"""

    text: str
    is_positive: bool
    embedding: Optional[torch.Tensor] = None


@dataclass
class CandidateSlot:
    """记录候选节点在局部子图中的排列，以及映射到的全局/额外节点索引。"""

    is_known: bool
    index: int


@dataclass
class ToolGraphSample:
    """标准化后的图样本，包含查询、候选工具及标签。"""

    query: str
    query_id: Any
    candidate_tool_ids: List[int]
    positive_tool_ids: Set[int]
    candidate_slots: List[CandidateSlot]
    extra_tools: List[ExtraToolNode]


class ToolGraphDataset(Dataset):
    """用于图神经网络训练/评估的Dataset，返回PyG Data对象。"""

    def __init__(
        self,
        samples: Sequence[ToolGraphSample],
        query_embeddings: torch.Tensor,
        tool_embeddings: torch.Tensor,
        adjacency: Dict[int, Dict[int, float]],
        *,
        device: str | None = None,
    ):
        if Data is None:  # pragma: no cover - 只有在未安装PyG时触发
            raise ImportError("torch_geometric未安装，无法构建图数据。请先安装torch_geometric后再使用gnn模型。")

        self.samples = list(samples)
        self.query_embeddings = query_embeddings.to(torch.float32)
        self.tool_embeddings = tool_embeddings.to(torch.float32)
        self.adjacency = adjacency
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.tool_embeddings.numel():
            self.feature_dim = self.tool_embeddings.shape[1]
        else:
            self.feature_dim = self.query_embeddings.shape[1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        query_embedding = self.query_embeddings[idx]
        tool_ids = sample.candidate_tool_ids
        extra_tools = sample.extra_tools
        candidate_slots = sample.candidate_slots

        if tool_ids:
            known_embeddings = self.tool_embeddings[torch.tensor(tool_ids, dtype=torch.long)]
        else:
            known_embeddings = torch.empty((0, self.feature_dim), dtype=torch.float32)
        if extra_tools:
            stacked = []
            for extra in extra_tools:
                if extra.embedding is None:
                    raise ValueError("额外工具节点缺少预计算的嵌入，请检查图数据预处理。")
                stacked.append(extra.embedding)
            extra_embeddings = torch.stack(stacked, dim=0).to(torch.float32)
        else:
            extra_embeddings = torch.empty((0, self.feature_dim), dtype=torch.float32)

        tool_embedding_list: List[torch.Tensor] = []
        labels = torch.zeros(len(candidate_slots) + 1, dtype=torch.float32)

        node_type = torch.zeros(len(candidate_slots) + 1, dtype=torch.long)
        node_type[1:] = 1  # 1表示工具节点
        tool_mask = node_type == 1

        positive_set = set(sample.positive_tool_ids)
        local_id_map: Dict[int, int] = {}
        tool_id_set: Set[int] = set()

        for local_idx, slot in enumerate(candidate_slots, start=1):
            if slot.is_known:
                global_tool_id = tool_ids[slot.index]
                tool_id_set.add(global_tool_id)
                local_id_map[global_tool_id] = local_idx
                embedding = known_embeddings[slot.index]
                if global_tool_id in positive_set:
                    labels[local_idx] = 1.0
            else:
                extra_tool = extra_tools[slot.index]
                embedding = extra_embeddings[slot.index]
                if extra_tool.is_positive:
                    labels[local_idx] = 1.0
            tool_embedding_list.append(embedding)

        if tool_embedding_list:
            tool_embeddings_tensor = torch.stack(tool_embedding_list, dim=0)
        else:
            tool_embeddings_tensor = torch.empty((0, self.feature_dim), dtype=torch.float32)

        query_embedding = query_embedding.unsqueeze(0)
        node_features = torch.cat([query_embedding, tool_embeddings_tensor], dim=0)

        edge_sources: List[int] = []
        edge_targets: List[int] = []
        edge_weights: List[float] = []

        # query节点与所有工具节点连接
        for local_idx in range(1, len(candidate_slots) + 1):
            edge_sources.extend([0, local_idx])
            edge_targets.extend([local_idx, 0])
            edge_weights.extend([1.0, 1.0])

        # 工具之间依据全局图连接
        for global_src in tool_id_set:
            neighbors = self.adjacency.get(global_src, {})
            for global_tgt, weight in neighbors.items():
                if global_tgt not in tool_id_set:
                    continue
                src_local = local_id_map[global_src]
                tgt_local = local_id_map[global_tgt]
                edge_sources.append(src_local)
                edge_targets.append(tgt_local)
                edge_weights.append(float(weight))

        if edge_sources:
            edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
            edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weight = torch.empty((0,), dtype=torch.float32)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=labels,
            node_type=node_type,
            tool_mask=tool_mask,
            num_nodes=node_features.shape[0],
        )
        if sample.query_id is not None:
            try:
                data.query_id = torch.tensor([int(sample.query_id)])
            except (TypeError, ValueError):
                pass  # 非数字id，仅在日志中使用
        return data


def _read_queries(file_path: Path) -> Dict[str, str]:
    df = pd.read_csv(file_path, sep="\t", names=["qid", "query"])
    return {row.qid: row.query for row in df.itertuples()}


def _read_qrels(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path, sep="\t", names=["qid", "useless", "docid", "label"])


def _build_relevant_docs(labels_df: pd.DataFrame) -> Dict[str, set]:
    relevant_docs: Dict[str, set] = {}
    for row in labels_df.itertuples():
        relevant_docs.setdefault(row.qid, set()).add(row.docid)
    return relevant_docs


def prepare_training_data(
    data_path: str | Path,
) -> Tuple[List[InputExample], Dict[str, str], Dict[str, str], Dict[str, set]]:
    data_dir = Path(data_path)
    documents_df = pd.read_csv(data_dir / "corpus.tsv", sep="\t")
    ir_corpus, _ = process_retrieval_ducoment(documents_df)

    train_queries = _read_queries(data_dir / "train.query.txt")
    test_queries = _read_queries(data_dir / "test.query.txt")

    train_labels_df = _read_qrels(data_dir / "qrels.train.tsv")
    train_samples: List[InputExample] = []
    for row in train_labels_df.itertuples():
        sample = InputExample(texts=[train_queries[row.qid], ir_corpus[row.docid]], label=row.label)
        train_samples.append(sample)

    test_labels_df = _read_qrels(data_dir / "qrels.test.tsv")
    relevant_docs = _build_relevant_docs(test_labels_df)
    return train_samples, test_queries, ir_corpus, relevant_docs


def load_evaluation_data(data_path: str | Path):
    data_dir = Path(data_path)
    documents_df = pd.read_csv(data_dir / "corpus.tsv", sep="\t")
    ir_corpus, _ = process_retrieval_ducoment(documents_df)
    test_queries = _read_queries(data_dir / "test.query.txt")
    test_labels_df = _read_qrels(data_dir / "qrels.test.tsv")
    relevant_docs = _build_relevant_docs(test_labels_df)
    return test_queries, ir_corpus, relevant_docs


# =====================
# GNN 数据准备辅助函数
# =====================


def _resolve_graph_split_file(base_path: Path, split_name: str, override: Optional[str]) -> Optional[Path]:
    if override:
        resolved = Path(override)
        if not resolved.exists():
            raise FileNotFoundError(f"未找到指定的{split_name}数据文件: {resolved}")
        return resolved

    if base_path.is_file():
        return base_path if split_name == "train" else None

    candidates = [
        base_path / f"{split_name}.jsonl",
        base_path / f"{split_name}.json",
        base_path / f"{split_name}.ndjson",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_tool_graph_records(file_path: Path) -> List[dict]:
    if file_path is None or not file_path.exists():
        return []

    with file_path.open("r", encoding="utf-8") as fp:
        if file_path.suffix in {".jsonl", ".ndjson"}:
            records = [json.loads(line) for line in fp if line.strip()]
        else:
            records = json.load(fp)

    if isinstance(records, dict):
        records = records.get("data", [])
    if not isinstance(records, list):
        raise ValueError(f"{file_path} 的内容格式必须是list或jsonl，每条为一个样本。")

    logging.info("读取%s，共%d条样本", file_path, len(records))
    return records


def _tool_key(tool_name: Optional[str], api_name: Optional[str]) -> Optional[str]:
    if not tool_name or not api_name:
        return None
    return f"{tool_name.strip()}::{api_name.strip()}".lower()


def _format_parameters(params: Optional[Iterable[dict]]) -> str:
    if not params:
        return ""
    formatted: List[str] = []
    for param in params:
        if not isinstance(param, dict):
            continue
        name = param.get("name", "param")
        p_type = param.get("type", "unk")
        description = param.get("description", "")
        default_value = param.get("default")
        suffix = []
        if description:
            suffix.append(str(description))
        if default_value not in (None, ""):
            suffix.append(f"default={default_value}")
        formatted.append(f"{name}({p_type}): {'; '.join(suffix)}".strip())
    return " | ".join(formatted)


def _build_tool_text(api_entry: dict) -> str:
    parts = []
    category = api_entry.get("category_name") or api_entry.get("category")
    tool_name = api_entry.get("tool_name") or api_entry.get("tool")
    api_name = api_entry.get("api_name") or api_entry.get("api")
    description = api_entry.get("api_description") or api_entry.get("description")
    method = api_entry.get("method")

    if category:
        parts.append(f"Category: {category}")
    if tool_name or api_name:
        parts.append(f"Tool: {tool_name or ''}::{api_name or ''}")
    if method:
        parts.append(f"Method: {method}")
    if description:
        parts.append(f"Desc: {description}")

    required = _format_parameters(api_entry.get("required_parameters"))
    optional = _format_parameters(api_entry.get("optional_parameters"))
    if required:
        parts.append(f"Required: {required}")
    if optional:
        parts.append(f"Optional: {optional}")

    template = api_entry.get("template_response")
    if template:
        parts.append(f"ResponseTemplate: {json.dumps(template, ensure_ascii=False)}")

    return " \n".join(parts)


def _collect_tool_metadata(records: Sequence[dict]) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for record in records:
        api_list = record.get("api_list") or record.get("apis") or []
        for api_entry in api_list:
            if not isinstance(api_entry, dict):
                continue
            key = _tool_key(api_entry.get("tool_name") or api_entry.get("tool"), api_entry.get("api_name") or api_entry.get("api"))
            if not key or key in metadata:
                continue
            metadata[key] = _build_tool_text(api_entry)
    return metadata


def _encode_texts(
    model: Qwen3Retriever,
    texts: Sequence[str],
    batch_size: int,
    *,
    expected_dim: Optional[int] = None,
) -> torch.Tensor:
    if not texts:
        if expected_dim is None:
            raise ValueError("无法在不知道维度的情况下创建空嵌入矩阵。")
        return torch.zeros((0, expected_dim), dtype=torch.float32)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)
    return embeddings.to(torch.float32).cpu()


def _assign_extra_tool_embeddings(
    samples: Sequence[ToolGraphSample],
    embedder: Qwen3Retriever,
    batch_size: int,
    expected_dim: int,
):
    extra_texts: List[str] = []
    owners: List[Tuple[int, int]] = []
    for sample_idx, sample in enumerate(samples):
        for extra_idx, extra in enumerate(sample.extra_tools):
            if extra.embedding is not None:
                continue
            extra_texts.append(extra.text)
            owners.append((sample_idx, extra_idx))

    if not extra_texts:
        return

    embeddings = _encode_texts(embedder, extra_texts, batch_size=batch_size, expected_dim=expected_dim)
    for tensor_idx, (sample_idx, extra_idx) in enumerate(owners):
        samples[sample_idx].extra_tools[extra_idx].embedding = embeddings[tensor_idx]


def _record_to_sample(record: dict, tool_to_id: Dict[str, int]) -> Optional[ToolGraphSample]:
    query_text = str(record.get("query") or "").strip()
    api_list = record.get("api_list") or record.get("apis") or []
    candidate_entries: List[Tuple[str, dict]] = []
    seen = set()
    for api_entry in api_list:
        if not isinstance(api_entry, dict):
            continue
        key = _tool_key(api_entry.get("tool_name") or api_entry.get("tool"), api_entry.get("api_name") or api_entry.get("api"))
        if key and key not in seen:
            candidate_entries.append((key, api_entry))
            seen.add(key)

    positive_entries = (
        record.get("relevant APIs")
        or record.get("relevant_APIs")
        or record.get("relevant_apis")
        or record.get("outputs")
        or []
    )
    positive_keys: Set[str] = set()
    for item in positive_entries:
        tool_name = None
        api_name = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            tool_name, api_name = item[0], item[1]
        elif isinstance(item, dict):
            tool_name = item.get("tool_name") or item.get("tool")
            api_name = item.get("api_name") or item.get("api")
        key = _tool_key(tool_name, api_name)
        if key:
            positive_keys.add(key)

    candidate_ids: List[int] = []
    extra_tools: List[ExtraToolNode] = []
    candidate_slots: List[CandidateSlot] = []
    candidate_key_set = {key for key, _ in candidate_entries}
    positive_keys &= candidate_key_set

    for key, api_entry in candidate_entries:
        if key in tool_to_id:
            idx = len(candidate_ids)
            candidate_ids.append(tool_to_id[key])
            candidate_slots.append(CandidateSlot(is_known=True, index=idx))
        else:
            node_text = _build_tool_text(api_entry)
            extra_tools.append(ExtraToolNode(text=node_text, is_positive=(key in positive_keys)))
            candidate_slots.append(CandidateSlot(is_known=False, index=len(extra_tools) - 1))

    positive_ids: Set[int] = {tool_to_id[key] for key in positive_keys if key in tool_to_id}

    has_positive = bool(positive_ids) or any(extra.is_positive for extra in extra_tools)
    if not candidate_slots or not has_positive:
        return None

    return ToolGraphSample(
        query=query_text,
        query_id=record.get("query_id"),
        candidate_tool_ids=candidate_ids,
        positive_tool_ids=positive_ids,
        candidate_slots=candidate_slots,
        extra_tools=extra_tools,
    )


def _count_tool_cooccurrences(samples: Sequence[ToolGraphSample], min_edge_frequency: int) -> Dict[int, Dict[int, float]]:
    edge_counter: Counter[Tuple[int, int]] = Counter()
    for sample in samples:
        positives = sorted(sample.positive_tool_ids)
        if len(positives) < 2:
            continue
        for src, tgt in combinations(positives, 2):
            edge_counter[(src, tgt)] += 1

    adjacency: Dict[int, Dict[int, float]] = defaultdict(dict)
    for (src, tgt), weight in edge_counter.items():
        if weight < min_edge_frequency:
            continue
        adjacency[src][tgt] = float(weight)
        adjacency[tgt][src] = float(weight)
    return adjacency


def _build_tool_graph_resources(
    records: Sequence[dict],
    *,
    min_edge_frequency: int,
    cache_path: Path,
    refresh_cache: bool,
    embedder: Qwen3Retriever,
    embedding_batch_size: int,
    cache_tag: str,
) -> Dict[str, Any]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh_cache:
        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict) and payload.get("_tag") == cache_tag:
            logging.info("加载缓存的工具图资源: %s", cache_path)
            return payload
        logging.info("缓存的工具图资源tag不匹配，重新构建: %s", cache_path)

    tool_metadata = _collect_tool_metadata(records)
    if not tool_metadata:
        raise ValueError("未能从数据中解析出任何工具节点，请检查api_list内容。")

    sorted_keys = sorted(tool_metadata.keys())
    tool_to_id = {key: idx for idx, key in enumerate(sorted_keys)}
    tool_texts = [tool_metadata[key] for key in sorted_keys]
    tool_embeddings = _encode_texts(embedder, tool_texts, batch_size=embedding_batch_size)

    payload = {
        "tool_to_id": tool_to_id,
        "tool_embeddings": tool_embeddings,
        "tool_keys": sorted_keys,
        "feature_dim": tool_embeddings.shape[1] if tool_embeddings.numel() else 0,
        "min_edge_frequency": min_edge_frequency,
        "_tag": cache_tag,
    }
    torch.save(payload, cache_path)
    logging.info("工具图资源已写入缓存: %s", cache_path)
    return payload


def prepare_tool_graph_data(
    data_path: str | Path,
    *,
    train_file: Optional[str] = None,
    eval_file: Optional[str] = None,
    test_file: Optional[str] = None,
    min_edge_frequency: int = 2,
    embedding_model_name: str = "Qwen/Qwen3-0.6B",
    embedding_batch_size: int = 4,
    cache_path: Optional[str] = None,
    refresh_cache: bool = False,
    device: Optional[str] = None,
) -> Tuple[Dict[str, Optional[ToolGraphDataset]], Dict[str, Any]]:
    base_path = Path(data_path)
    split_files = {
        "train": _resolve_graph_split_file(base_path, "train", train_file),
        "eval": _resolve_graph_split_file(base_path, "eval", eval_file),
        "test": _resolve_graph_split_file(base_path, "test", test_file),
    }

    raw_splits: Dict[str, List[dict]] = {}
    for split, file_path in split_files.items():
        if file_path is None:
            continue
        raw_splits[split] = _load_tool_graph_records(file_path)

    if not raw_splits:
        raise ValueError("未找到任何图数据文件，请使用--graph_*参数或确保目录下存在train/eval/test json。")

    train_records = list(raw_splits.get("train", []))
    if not train_records:
        raise ValueError("训练split为空或未提供，无法构建图工具全集。")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Qwen3Retriever(model_name_or_path=embedding_model_name, device=device)

    cache_file = Path(cache_path) if cache_path else base_path / "tool_graph_cache.pt"
    graph_resources = _build_tool_graph_resources(
        train_records,
        min_edge_frequency=min_edge_frequency,
        cache_path=cache_file,
        refresh_cache=refresh_cache,
        embedder=embedder,
        embedding_batch_size=embedding_batch_size,
        cache_tag=_TOOL_VOCAB_CACHE_TAG,
    )

    tool_to_id = graph_resources["tool_to_id"]
    tool_embeddings = graph_resources["tool_embeddings"]
    feature_dim = graph_resources.get("feature_dim") or (tool_embeddings.shape[1] if tool_embeddings.numel() else None)
    if feature_dim is None:
        raise ValueError("无法确定工具嵌入维度，请检查工具嵌入构建。")

    split_samples: Dict[str, List[ToolGraphSample]] = {}
    all_samples: List[ToolGraphSample] = []
    for split, records in raw_splits.items():
        samples = []
        for record in records:
            sample = _record_to_sample(record, tool_to_id)
            if sample is not None:
                samples.append(sample)
                all_samples.append(sample)
        split_samples[split] = samples
        logging.info("%s split 保留%d条样本（过滤掉候选或标签缺失的数据）", split, len(samples))

    train_samples = split_samples.get("train", [])
    if not train_samples:
        raise ValueError("训练split没有可用样本，请检查是否包含合法的候选与正例。")

    if not all_samples:
        raise ValueError("所有样本都被过滤掉了，请检查数据内容是否包含api_list和relevant APIs。")

    _assign_extra_tool_embeddings(all_samples, embedder, embedding_batch_size, expected_dim=feature_dim)

    adjacency = _count_tool_cooccurrences(train_samples, min_edge_frequency=min_edge_frequency)
    adjacency = {src: dict(neigh) for src, neigh in adjacency.items()}
    num_edges = sum(len(neigh) for neigh in adjacency.values()) // 2
    logging.info("大图节点数=%d, 边数(无向)=%d", len(tool_to_id), num_edges)

    dataset_cache_file = base_path / "graph_dataset_cache.pt"
    datasets: Optional[Dict[str, Optional[ToolGraphDataset]]] = None
    if dataset_cache_file.exists() and not refresh_cache:
        payload = torch.load(dataset_cache_file, map_location="cpu")
        if isinstance(payload, dict) and payload.get("_tag") == _GRAPH_DATASET_CACHE_TAG:
            logging.info("加载缓存的PyG Dataset: %s", dataset_cache_file)
            cached_splits = payload.get("splits", {})
            datasets = {split: cached_splits.get(split) for split in ("train", "eval", "test")}
        else:
            logging.info("PyG Dataset缓存tag不匹配，重新构建: %s", dataset_cache_file)

    if datasets is None:
        queries = [sample.query for sample in all_samples]
        query_embeddings_all = _encode_texts(
            embedder,
            queries,
            batch_size=embedding_batch_size,
            expected_dim=feature_dim,
        )

        datasets: Dict[str, Optional[ToolGraphDataset]] = {}
        offset = 0
        for split, samples in split_samples.items():
            count = len(samples)
            if count == 0:
                datasets[split] = None
                continue
            split_query_embeddings = query_embeddings_all[offset : offset + count]
            datasets[split] = ToolGraphDataset(
                samples=samples,
                query_embeddings=split_query_embeddings,
                tool_embeddings=tool_embeddings,
                adjacency=adjacency,
                device=device,
            )
            offset += count

        payload = {"_tag": _GRAPH_DATASET_CACHE_TAG, "splits": datasets}
        torch.save(payload, dataset_cache_file)
        logging.info("PyG Dataset缓存已写入: %s", dataset_cache_file)

    id_to_tool = {idx: key for key, idx in tool_to_id.items()}
    graph_meta = {
        "feature_dim": feature_dim,
        "tool_to_id": tool_to_id,
        "id_to_tool": id_to_tool,
        "num_edges": num_edges,
        "num_nodes": len(tool_to_id),
    }

    return datasets, graph_meta
