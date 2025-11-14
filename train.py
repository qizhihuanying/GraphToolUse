#!/usr/bin/env python3
"""
gpu_manager.py

每隔 interval 秒扫描指定 GPU 列表（默认 0-7）。
若某张 GPU 的 空闲显存 (free) >= threshold (GB)，且该卡当前没有正在运行的 worker，
则为该 GPU 启动一个 worker 进程，worker 会占用指定显存并进行矩阵乘法循环计算（可用 Ctrl+C 停止所有进程）。

示例:
    python gpu_manager.py --gpus 0 1 2 3 4 5 6 7 --n 4096 --mem 20 --threshold 20 --interval 10
"""
import argparse
import multiprocessing as mp
import subprocess
import shlex
import time
import os
import signal
import sys
import random
import torch

def query_gpu_free_mem(gpus):
    """
    使用 nvidia-smi 查询给定 GPU 列表的 total/free（单位 MB），
    返回 dict: {gpu_index: (total_mb, free_mb)}。
    若查询失败则抛出 RuntimeError。
    """
    # 格式化为只返回 memory.total and memory.free（单位 MB）
    # 使用 comma-separated query 并对应每行
    cmd = "nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,nounits,noheader"
    try:
        out = subprocess.check_output(shlex.split(cmd), encoding="utf-8")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi 查询失败: {e}") from e

    res = {}
    for line in out.strip().splitlines():
        # each line: index, total, free
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            total_mb = int(parts[1])
            free_mb = int(parts[2])
        except ValueError:
            continue
        if idx in gpus:
            res[idx] = (total_mb, free_mb)
    return res

def worker(gpu, n, mem_gb):
    # 与你原先的 worker 基本一致（带中文输出）
    try:
        torch.cuda.set_device(gpu)
    except Exception as e:
        print(f"[GPU{gpu}] 无法 set_device: {e}", flush=True)
        return

    device = torch.device(f"cuda:{gpu}")
    try:
        name = torch.cuda.get_device_name(device)
        total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    except Exception as e:
        print(f"[GPU{gpu}] 无法读取设备信息: {e}", flush=True)
        return

    # 为该 GPU 生成 ±0.5GB 的浮动
    mem_gb_eff = mem_gb + random.uniform(-0.5, 0.5)
    # 预留 ~1GB 防止 OOM，且不小于 0.1GB
    mem_gb_eff = max(0.1, min(mem_gb_eff, max(0.0, total_gb - 1.0)))

    print(f"[GPU{gpu}] {name} | n={n} | target_mem≈{mem_gb_eff:.2f}GB (base {mem_gb:.2f}GB ±0.5GB)", flush=True)

    # 占显存（fp32）
    bytes_per_elem = 4
    target_bytes = int(mem_gb_eff * 1024**3)
    elems = max(1, target_bytes // bytes_per_elem)
    try:
        buf = torch.empty(elems, dtype=torch.float32, device=device)
        buf.fill_(1.0)  # 触发实际分配
    except RuntimeError as e:
        print(f"[GPU{gpu}] 显存分配失败: {e}", flush=True)
        return

    # 生成计算矩阵
    try:
        A = torch.randn((n, n), device=device, dtype=torch.float32)
        B = torch.randn((n, n), device=device, dtype=torch.float32)
    except RuntimeError as e:
        print(f"[GPU{gpu}] 矩阵分配失败: {e}", flush=True)
        del buf
        torch.cuda.empty_cache()
        return

    stop = mp.Event()

    def _signal_handler(signum, frame):
        # set stop flag
        stop.set()

    # 信号处理（在子进程内）
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # 热身
    try:
        for _ in range(3):
            _ = A @ B
            torch.cuda.synchronize()
    except Exception as e:
        print(f"[GPU{gpu}] 计算热身失败: {e}", flush=True)
        stop.set()

    print(f"[GPU{gpu}] 开始计算（进程 {os.getpid()}，按 Ctrl+C 停止全部）", flush=True)
    try:
        while not stop.is_set():
            _ = A @ B
            torch.cuda.synchronize()
    except Exception as e:
        print(f"[GPU{gpu}] 计算中异常: {e}", flush=True)

    # 清理
    try:
        del buf, A, B
        torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[GPU{gpu}] 已释放显存并退出（进程 {os.getpid()}）", flush=True)

def spawn_worker_process(gpu, n, mem_gb):
    """在一个新进程中运行 worker（multiprocessing.Process）"""
    p = mp.Process(target=worker, args=(gpu, n, mem_gb), daemon=False)
    p.start()
    return p

def main():
    parser = argparse.ArgumentParser(description="GPU 扫描器：每隔 interval 秒扫描指定 GPU，若 free >= threshold(GB) 则启动占用 worker")
    parser.add_argument("--gpus", type=int, nargs="+", default=[6,7], help="要扫描的 GPU 列表，默认为 0-7")
    parser.add_argument("--n", type=int, default=4096, help="矩阵大小，例如 4096 或 8192")
    parser.add_argument("--mem", type=float, default=20.0, help="每张卡基础占用显存（GB），将施加±0.5GB浮动")
    parser.add_argument("--threshold", type=float, default=20.0, help="触发启动的空闲显存阈值（GB），如果 free >= threshold 则启动")
    parser.add_argument("--interval", type=int, default=2, help="扫描间隔（秒），默认 10s")
    args = parser.parse_args()

    # ensure list of ints (in case default used)
    gpus = list(dict.fromkeys(args.gpus))  # 去重，保持顺序
    print(f"管理器启动 — 扫描 GPUs {gpus} | n={args.n} | base_mem={args.mem}GB ±0.5GB | threshold={args.threshold}GB | interval={args.interval}s")

    # 存放每个 gpu 的进程（index -> Process）
    procs = {}

    # 全局信号处理： Ctrl+C 后优雅退出所有子进程
    stop_main = False
    def _main_sig_handler(signum, frame):
        nonlocal stop_main
        stop_main = True
    signal.signal(signal.SIGINT, _main_sig_handler)
    signal.signal(signal.SIGTERM, _main_sig_handler)

    try:
        while not stop_main:
            try:
                stats = query_gpu_free_mem(gpus)
            except Exception as e:
                print(f"[manager] 无法查询 GPU 状态: {e}", flush=True)
                stats = {}

            for gpu in gpus:
                info = stats.get(gpu)
                if info is None:
                    # 未查询到该卡信息（可能不存在或 nvidia-smi 返回被过滤）
                    continue
                total_mb, free_mb = info
                total_gb = total_mb / 1024.0
                free_gb = free_mb / 1024.0

                # 若 free >= threshold 且该卡没有正在运行的 worker，就启动
                if free_gb >= args.threshold and gpu not in procs:
                    # 为了避免直接使用用户请求的 mem 导致 OOM，把 mem 限制为 free_gb - 1.0（保留 1GB）
                    mem_to_use = min(args.mem, max(0.1, free_gb - 1.0))
                    print(f"[manager] GPU{gpu} 空闲 {free_gb:.2f}GB (total {total_gb:.2f}GB) >= {args.threshold}GB -> 启动 worker (mem={mem_to_use:.2f}GB)", flush=True)
                    p = spawn_worker_process(gpu, args.n, mem_to_use)
                    procs[gpu] = p
                else:
                    # 输出当前状态（仅做观察）
                    print(f"[manager] GPU{gpu}: free={free_gb:.2f}GB / total={total_gb:.2f}GB", flush=True)

            # 清理已退出的进程条目
            to_remove = []
            for g, p in procs.items():
                if not p.is_alive():
                    print(f"[manager] GPU{g} 的 worker (pid={p.pid}) 已退出，移除记录。", flush=True)
                    to_remove.append(g)
            for g in to_remove:
                procs.pop(g, None)

            # sleep until next scan or interrupted
            for _ in range(args.interval):
                if stop_main:
                    break
                time.sleep(1)
    except Exception as e:
        print(f"[manager] 异常退出: {e}", file=sys.stderr)
    finally:
        print("[manager] 终止中：停止所有子进程...", flush=True)
        # try graceful terminate
        for g, p in procs.items():
            try:
                if p.is_alive():
                    print(f"[manager] 终止 GPU{g} 的 worker (pid={p.pid})", flush=True)
                    p.terminate()
            except Exception as e:
                print(f"[manager] 无法终止 GPU{g} 的 worker: {e}", flush=True)
        # 等待短时间让它们退出
        timeout = 5
        end = time.time() + timeout
        while time.time() < end and any(p.is_alive() for p in procs.values()):
            time.sleep(0.1)
        # 强制 kill (如果仍然存在) - 注意：multiprocessing.Process 没有 kill 跨平台方法，这里用 terminate 再检查
        for g, p in procs.items():
            if p.is_alive():
                print(f"[manager] worker (pid={p.pid}) 仍未退出，尝试 terminate 再次...", flush=True)
                try:
                    p.terminate()
                except Exception:
                    pass
        print("[manager] 完成清理，退出。", flush=True)

if __name__ == "__main__":
    # 在主进程设置 start method
    mp.set_start_method("spawn", force=True)
    main()
