import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models, InputExample, losses, LoggingHandler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from api_evaluator import APIEvaluator
from toolbench.utils import process_retrieval_ducoment

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "log"
logger = logging.getLogger(__name__)
_PRINTER_CALLBACK_PATCHED = False

class _DeviceNameFilter(logging.Filter):
    def filter(self, record):
        return "Use pytorch device_name" not in record.getMessage()


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default=None, type=str, required=True,
                    help="The input data dir. Should contain the .tsv files for the task.")
parser.add_argument("--model_name", default=None, type=str, required=True,
                    help="The base model name.")
parser.add_argument("--output_path", default=None, type=str, required=True,
                    help="The base path where the model output will be saved.")
parser.add_argument("--num_epochs", default=5, type=int, required=True,
                    help="Train epochs.")
parser.add_argument("--train_batch_size", default=32, type=int, required=True,
                    help="Train batch size.")
parser.add_argument("--learning_rate", default=2e-5, type=float, required=True,
                    help="Learning rate.")
parser.add_argument("--warmup_steps", default=500, type=float, required=True,
                    help="Warmup steps.")
parser.add_argument("--max_seq_length", default=256, type=int, required=True,
                    help="Max sequence length.")
parser.add_argument("--gpu_id", default=None, type=str, required=False,
                    help="Comma-separated CUDA device id(s) to use, e.g. '0' or '0,1'.")
parser.add_argument("--log_path", default=None, type=str, required=False,
                    help="Directory where training logs will be stored. Defaults to <repo>/log.")


def patch_trainer_logging_callbacks():
    global _PRINTER_CALLBACK_PATCHED
    if _PRINTER_CALLBACK_PATCHED:
        return

    original_printer_on_log = PrinterCallback.on_log
    original_progress_on_log = ProgressCallback.on_log

    def patched_printer_on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return original_printer_on_log(self, args, state, control, logs, **kwargs)
        logs_copy = dict(logs)
        result = original_printer_on_log(self, args, state, control, logs, **kwargs)
        if getattr(args, "disable_tqdm", False) and getattr(state, "is_local_process_zero", True):
            logs_copy.pop("total_flos", None)
            logging.info(json.dumps(logs_copy, ensure_ascii=False))
        return result

    def patched_progress_on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return original_progress_on_log(self, args, state, control, logs, **kwargs)
        logs_copy = dict(logs)
        result = original_progress_on_log(self, args, state, control, logs, **kwargs)
        if not getattr(args, "disable_tqdm", False) and getattr(state, "is_world_process_zero", True):
            logs_copy.pop("total_flos", None)
            logging.info(json.dumps(logs_copy, ensure_ascii=False))
        return result

    PrinterCallback.on_log = patched_printer_on_log
    ProgressCallback.on_log = patched_progress_on_log
    _PRINTER_CALLBACK_PATCHED = True


def prepare_log_file(args):
    log_dir = Path(args.log_path).expanduser() if args.log_path else DEFAULT_LOG_DIR
    if not log_dir.is_absolute():
        log_dir = (REPO_ROOT / log_dir).resolve()
    else:
        log_dir = log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    lr_str = f"{args.learning_rate:g}"
    log_file_name = f"lr={lr_str}+bs={args.train_batch_size}+epoch={args.num_epochs}.log"
    log_file_path = log_dir / log_file_name

    params_line = json.dumps(vars(args), ensure_ascii=False, sort_keys=True)
    log_file_path.write_text(params_line + "\n", encoding="utf-8")
    return log_file_path


def configure_logging(log_file_path: Path):
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s',
                                                datefmt='%Y-%m-%d %H:%M:%S'))
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler(), file_handler])
    st_logger = logging.getLogger("sentence_transformers.SentenceTransformer")
    st_logger.addFilter(_DeviceNameFilter())


def main():
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    log_file_path = prepare_log_file(args)
    configure_logging(log_file_path)
    patch_trainer_logging_callbacks()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    lr = args.learning_rate
    warmup_steps = args.warmup_steps
    data_path = args.data_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    model_save_path = os.path.join(output_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(model_save_path, exist_ok=True)

    tensorboard_name = 'name_desc'
    logs_writer = SummaryWriter(os.path.join(output_path, 'tensorboard', tensorboard_name))

    def log_callback_st(train_ix, global_step, training_steps, current_lr, loss_value):
        logs_writer.add_scalar('train_loss', loss_value, global_step)
        logs_writer.add_scalar('lr', current_lr[0], global_step)

    # Model definition
    word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ir_train_queries = {}
    ir_test_queries = {}
    ir_relevant_docs = {}
    train_samples = []

    documents_df = pd.read_csv(os.path.join(data_path, 'corpus.tsv'), sep='\t')
    ir_corpus, _ = process_retrieval_ducoment(documents_df)

    train_queries_df = pd.read_csv(os.path.join(data_path, 'train.query.txt'), sep='\t', names=['qid', 'query'])
    for row in train_queries_df.itertuples():
        ir_train_queries[row.qid] = row.query
    train_queries_df = pd.read_csv(os.path.join(data_path, 'test.query.txt'), sep='\t', names=['qid', 'query'])
    for row in train_queries_df.itertuples():
        ir_test_queries[row.qid] = row.query

    labels_df = pd.read_csv(os.path.join(data_path, 'qrels.train.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
    for row in labels_df.itertuples():
        sample = InputExample(texts=[ir_train_queries[row.qid], ir_corpus[row.docid]], label=row.label)
        train_samples.append(sample)
    labels_df = pd.read_csv(os.path.join(data_path, 'qrels.test.tsv'), sep='\t', names=['qid', 'useless', 'docid', 'label'])
    for row in labels_df.itertuples():
        ir_relevant_docs.setdefault(row.qid, set()).add(row.docid)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, pin_memory=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    ir_evaluator = APIEvaluator(ir_test_queries, ir_corpus, ir_relevant_docs)

    # You may need to modify the .fit() method to ensure all data is moved to the correct device during parallel computations

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=ir_evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': lr},
        output_path=model_save_path,
    )

    logs_writer.close()


if __name__ == "__main__":
    main()
