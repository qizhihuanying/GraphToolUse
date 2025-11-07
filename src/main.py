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

from data_loader import prepare_training_data
from train import APIEvaluator, Trainer, get_model
from utils import DeviceNameFilter, configure_logging, prepare_log_file


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Directory containing retrieval TSV files.")
    parser.add_argument("--model_type", type=str, default="sentence_transformer",
                        help="Registered model type to use for training.")
    parser.add_argument("--model_name_or_path", "--model_name", dest="model_name_or_path",
                        type=str, required=True, help="Base model name or path.")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save checkpoints.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length.")
    parser.add_argument("--torch_dtype", type=str, default="auto", help="Torch dtype hint for decoder-only models.")
    parser.add_argument("--gpu_id", type=str, default=None,
                        help="CUDA device ids, e.g. '0' or '0,1'.")
    parser.add_argument("--log_path", type=str, default=None,
                        help="Directory where training logs will be stored. Defaults to <repo>/log.")
    parser.add_argument("--torch_seed", type=int, default=42, help="Random seed for torch.")
    parser.add_argument("--results_path", type=str, default=None, help="Optional JSON file to store metrics.")
    return parser


def build_log_file_name(args) -> str:
    lr_str = f"{args.learning_rate:g}"
    return f"lr={lr_str}+bs={args.train_batch_size}+epoch={args.num_epochs}.log"


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    log_file_path = prepare_log_file(build_log_file_name(args), vars(args), args.log_path)
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
    model = get_model(model_config)

    train_samples, test_queries, ir_corpus, ir_relevant_docs = prepare_training_data(args.data_path)
    dataloader = None
    if args.num_epochs > 0 and getattr(model, "is_trainable", False):
        dataloader = DataLoader(
            train_samples,
            shuffle=True,
            batch_size=args.train_batch_size,
            pin_memory=torch.cuda.is_available(),
        )

    evaluator = APIEvaluator(
        test_queries,
        ir_corpus,
        ir_relevant_docs,
        batch_size=args.train_batch_size,
    )

    tensorboard_dir = output_root / "tensorboard" / Path(args.model_name_or_path).name.replace("/", "_")
    if args.num_epochs > 0:
        eval_evaluator = evaluator
        test_evaluator = None
    else:
        eval_evaluator = None
        test_evaluator = evaluator

    results_path = Path(args.results_path) if args.results_path else None
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
