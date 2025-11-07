# Repository Guidelines

## Project Structure & Module Organization
- `toolbench/`: Core Python packages for retrieval; `retrieval/train.py` orchestrates SentenceTransformer fine-tuning and `api_evaluator.py` scores runs.
- `scripts/`: Bash wrappers such as `train_retriever.sh` and `preprocess_retriever_data.sh`; keep them executable and parameterize via environment variables.
- `preprocess/`: Data shaping utilities that create corpus/query TSVs before training.
- `data/`: Local datasets (`instruction/`, `retrieval/`, etc.); keep large raw assets out of commits and document any external download steps.
- `retriever_model/` and `checkpoints/`: Generated artifacts; prune stale runs and avoid tracking them in version control.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: Install Python ≥3.9 dependencies; run inside a virtualenv or Conda environment.
- `bash scripts/preprocess_retriever_data.sh`: Regenerate `data/retrieval/<split>` corpora; rerun when queries or indices change.
- `bash scripts/train_retriever.sh`: Train the retriever with the configured `GPU_ID`, emitting timestamped folders under `retriever_model/`.
- `python toolbench/retrieval/inference_example.py <model_dir> data/retrieval/G3`: Spot-check retrieval quality and produce `top5_results_with_matches.json`.

## Coding Style & Naming Conventions
- Python modules follow PEP 8: 4-space indentation, `snake_case` functions, `CamelCase` classes, UPPER_SNAKE constants.
- Prefer explicit type-aware helpers (e.g., `Path`, `dataclass`) and keep docstrings brief but action-oriented.
- Maintain deterministic behavior—seed Torch (`torch.manual_seed`) when adding new entry points.
- New scripts should accept flags via `argparse` and default to repo-relative paths so they run from the project root.

## Testing Guidelines
- There is no formal test suite; rely on scripted evaluations. Always rerun `inference_example.py` using the latest checkpoint and compare `successful_matches`.
- For preprocessing changes, diff generated TSV/TXT counts (`wc -l`) to confirm splits are stable.
- When touching evaluator logic, run a short training epoch (`--num_epochs 1`) and ensure `log_file.txt` and TensorBoard scalars remain sane.
- Document any manual verification in the pull request, especially when metrics or artifacts change.

## Commit & Pull Request Guidelines
- Write imperative, concise commit titles (e.g., `Add GPU selector to retriever training`); include context in the body when touching data or configs.
- Pull requests should summarize intent, enumerate runnable commands, link related issues, and attach before/after metrics or screenshots when applicable.
- Flag new dependencies in both `requirements.txt` and the PR description, and mention whether regeneration of `data/` or `retriever_model/` artifacts is required.

## Security & Configuration Tips
- Keep API credentials (e.g., RapidAPI keys) in environment variables or ignored config files—never commit secrets.
- Use the `--gpu_id` flag or `CUDA_VISIBLE_DEVICES` to constrain GPU usage on shared machines.
- Large generated logs (`tensorboard/`, `top5_results_with_matches.json`) should be cleaned or gitignored before opening a PR.

# Always respond in Chinese-simplified.
