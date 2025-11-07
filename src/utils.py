import json
import logging
import re
from datetime import datetime
from pathlib import Path
from functools import partial

import torch
import transformers
import transformers.models.llama.modeling_llama
from sentence_transformers import LoggingHandler


def process_system_message(system_message, functions):
    assert "with a function call to actually excute your step." in system_message
    # we find that following ReACT format and merging the thought node and function call node is easier for model to learn to integrate the action input json string in its prediction than learn to predict a json string directly.
    system_message = system_message.replace("with a function call to actually excute your step.", "with a function call to actually excute your step. Your output should follow this format:\nThought:\nAction\nAction Input:\n")
    # add all the function dicts in the prompt.
    system_message = system_message + "\nSpecifically, you have access to the following APIs: " + str(functions)
    return system_message

def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def standardize_category(category):
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category

def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string

def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name

# code adapted from https://huggingface.co/kaiokendev/superhot-13b-8k-no-rlhf-test/blob/main/llama_rope_scaled_monkey_patch.py
class CondenseRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, ratio, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build here to make `torch.jit.trace` work.
        self.ratio = ratio
        max_position_embeddings *= ratio
        print(f"Condensing Positional embeddings from {max_position_embeddings} to {max_position_embeddings // ratio}")
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype) / ratio
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype) / self.ratio
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def replace_llama_with_condense(ratio):
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = partial(CondenseRotaryEmbedding, ratio=ratio)

    
def process_retrieval_ducoment(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = json.loads(row.document_content)
        ir_corpus[row.docid] = (doc.get('category_name', '') or '') + ', ' + \
        (doc.get('tool_name', '') or '') + ', ' + \
        (doc.get('api_name', '') or '') + ', ' + \
        (doc.get('api_description', '') or '') + \
        ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
        ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
        ', return_schema: ' + json.dumps(doc.get('template_response', ''))
        corpus2tool[(doc.get('category_name', '') or '') + ', ' + \
        (doc.get('tool_name', '') or '') + ', ' + \
        (doc.get('api_name', '') or '') + ', ' + \
        (doc.get('api_description', '') or '') + \
        ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
        ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) + \
        ', return_schema: ' + json.dumps(doc.get('template_response', ''))] = doc['category_name'] + '\t' + doc['tool_name'] + '\t' + doc['api_name']
    return ir_corpus, corpus2tool


# Logging helpers

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = REPO_ROOT / "log"


class DeviceNameFilter(logging.Filter):
    _BLOCKED_MSGS = [
        "Use pytorch device_name",
        "No sentence-transformers model found with name",
    ]

    def filter(self, record):
        message = record.getMessage()
        return not any(block in message for block in self._BLOCKED_MSGS)


def resolve_log_directory(log_dir: str | Path | None) -> Path:
    if log_dir:
        path = Path(log_dir).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
    else:
        path = DEFAULT_LOG_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_log_outputs(run_name: str, params: dict, log_dir: str | Path | None = None):
    base_dir = resolve_log_directory(log_dir) / run_name
    base_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = base_dir / "run.log"
    results_path = base_dir / "results.json"
    separator = "=" * 80
    needs_newline = log_file_path.exists() and log_file_path.stat().st_size > 0
    with log_file_path.open("a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(f"{separator}\nRun started at {datetime.now().isoformat()}\n")
        if params:
            f.write(json.dumps(params, ensure_ascii=False, sort_keys=True) + "\n")
    return log_file_path, results_path


def configure_logging(log_file_path: Path, extra_filters=None):
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    console_handler = LoggingHandler()
    console_handler.setFormatter(formatter)

    handlers = [console_handler, file_handler]
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.INFO)
    for handler in handlers:
        root_logger.addHandler(handler)

    if extra_filters:
        for handler in handlers:
            for filt in extra_filters:
                handler.addFilter(filt)
    
