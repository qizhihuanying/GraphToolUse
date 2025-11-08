from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure project root available when running as `python src/main.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import prepare_tool_graph_data, prepare_training_data
from train import APIEvaluator, Trainer, get_model
from utils import DeviceNameFilter, configure_logging, prepare_log_outputs


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Directory containing retrieval TSV files.")
    parser.add_argument("--model_type", type=str, default="sentence_transformer",
                        help="Registered model type to use for training.")
    parser.add_argument("--model_name_or_path", "--model_name", dest="model_name_or_path",
                        type=str, required=True, help="Base model name or path.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Training/evaluation batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length.")
    parser.add_argument("--torch_dtype", type=str, default="auto", help="Torch dtype hint for decoder-only models.")
    parser.add_argument("--gpu_id", type=str, default=None,
                        help="CUDA device ids, e.g. '0' or '0,1'.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Directory where training logs will be stored. Defaults to <repo>/log.")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch.")
    parser.add_argument("--gnn_hidden_dim", type=int, default=768, help="QueryAwareGNN隐藏层维度。")
    parser.add_argument("--gnn_layers", type=int, default=2, help="GCN层数。")
    parser.add_argument("--gnn_weight_decay", type=float, default=0.01, help="GNN优化器的weight decay。")
    return parser


def build_log_file_name(args) -> str:
    lr_str = f"{args.learning_rate:g}"
    return f"lr={lr_str}+bs={args.batch_size}+epoch={args.num_epochs}"


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    log_file_path, results_path = prepare_log_outputs(build_log_file_name(args), vars(args), args.log_path)
    configure_logging(log_file_path, extra_filters=[DeviceNameFilter()])

    torch.manual_seed(args.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)

    output_root = Path(args.output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    model_save_path = output_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = {
        "model_type": args.model_type,
        "model_name_or_path": args.model_name_or_path,
        "max_seq_length": args.max_seq_length,
        "device": device,
        "torch_dtype": args.torch_dtype,
    }

    graph_datasets = None
    graph_meta = None
    if args.model_type == "gnn":
        GRAPH_EMBEDDING_MODEL = "Qwen/Qwen3-0.6B"
        GRAPH_EMBED_BATCH_SIZE = 32
        GRAPH_MIN_EDGE_FREQ = 2
        graph_datasets, graph_meta = prepare_tool_graph_data(
            args.data_path,
            train_file=None,
            eval_file=None,
            test_file=None,
            min_edge_frequency=GRAPH_MIN_EDGE_FREQ,
            embedding_model_name=GRAPH_EMBEDDING_MODEL,
            embedding_batch_size=GRAPH_EMBED_BATCH_SIZE,
            cache_path=None,
            refresh_cache=False,
            device=device,
        )
        feature_dim = graph_meta.get("feature_dim")
        if not feature_dim:
            raise ValueError("未能确定图节点特征维度，请检查图预处理。")
        model_config.update(
            {
                "input_dim": feature_dim,
                "hidden_dim": args.gnn_hidden_dim,
                "gnn_layers": args.gnn_layers,
                "dropout_rate": 0.1,
                "pos_weight": 1.0,
                "weight_decay": args.gnn_weight_decay,
            }
        )

    model = get_model(model_config)

    dataloader = None
    evaluator = None
    eval_evaluator = None
    test_evaluator = None

    if args.model_type == "gnn":
        from torch_geometric.loader import DataLoader as GraphDataLoader

        train_dataset = graph_datasets.get("train") if graph_datasets else None
        if args.num_epochs > 0:
            if train_dataset is None:
                raise ValueError("未找到图训练数据，无法执行训练。")
            dataloader = GraphDataLoader(
                train_dataset,
                shuffle=True,
                batch_size=args.batch_size,
            )

        eval_dataset = graph_datasets.get("eval") if graph_datasets else None
        test_dataset = graph_datasets.get("test") if graph_datasets else None

        eval_evaluator_candidate = (
            APIEvaluator(
                 graph_dataset=eval_dataset,
                 graph_batch_size=args.batch_size,
                 device=device,
            )
            if eval_dataset is not None
            else None
        )
        test_evaluator_candidate = (
            APIEvaluator(
                graph_dataset=test_dataset,
                graph_batch_size=args.batch_size,
                device=device,
            )
            if test_dataset is not None
            else None
        )

        if args.num_epochs > 0:
            eval_evaluator = eval_evaluator_candidate or test_evaluator_candidate
            test_evaluator = None
        else:
            eval_evaluator = None
            test_evaluator = test_evaluator_candidate or eval_evaluator_candidate
    else:
        train_samples, test_queries, ir_corpus, ir_relevant_docs = prepare_training_data(args.data_path)
        if args.num_epochs > 0 and getattr(model, "is_trainable", False):
            dataloader = DataLoader(
                train_samples,
                shuffle=True,
                batch_size=args.batch_size,
                pin_memory=torch.cuda.is_available(),
            )

        evaluator = APIEvaluator(
            test_queries,
            ir_corpus,
            ir_relevant_docs,
            batch_size=args.batch_size,
        )

        if args.num_epochs > 0:
            eval_evaluator = evaluator
            test_evaluator = None
        else:
            eval_evaluator = None
            test_evaluator = evaluator

    tensorboard_dir = output_root / "tensorboard" / Path(args.model_name_or_path).name.replace("/", "_")

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        evaluator=eval_evaluator,
        test_evaluator=test_evaluator,
        output_path=str(model_save_path),
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        tensorboard_dir=tensorboard_dir,
        results_path=results_path,
    )

    trainer.run()


if __name__ == "__main__":
    main()
