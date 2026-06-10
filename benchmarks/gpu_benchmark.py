#!/usr/bin/env python3
import gc
import json
import math
import time

import torch


def sync():
    torch.cuda.synchronize()


def event_elapsed_ms(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    sync()
    return start.elapsed_time(end) / reps


def bench_matmul(n, dtype, allow_tf32=None, reps=10):
    if allow_tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    a = torch.randn((n, n), device="cuda", dtype=dtype)
    b = torch.randn((n, n), device="cuda", dtype=dtype)
    c = torch.empty((n, n), device="cuda", dtype=dtype)

    def fn():
        torch.matmul(a, b, out=c)

    ms = event_elapsed_ms(fn, warmup=3, reps=reps)
    tflops = (2 * n**3) / (ms / 1000) / 1e12
    result = {
        "kind": "matmul",
        "n": n,
        "dtype": str(dtype).replace("torch.", ""),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "ms": round(ms, 3),
        "tflops": round(tflops, 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    del a, b, c
    torch.cuda.empty_cache()
    gc.collect()
    return result


def bench_copy(gib=2, reps=20):
    elems = int(gib * 1024**3 // 2)
    a = torch.empty((elems,), device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)
    a.normal_()

    def fn():
        b.copy_(a)

    ms = event_elapsed_ms(fn, warmup=3, reps=reps)
    gbps = (gib * 1024**3) / (ms / 1000) / 1e9
    result = {
        "kind": "device_copy",
        "gib": gib,
        "ms": round(ms, 3),
        "gbps": round(gbps, 3),
        "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }
    del a, b
    torch.cuda.empty_cache()
    gc.collect()
    return result


def bench_memory_capacity(chunk_gib=1.0, safety_gib=1.5):
    torch.cuda.empty_cache()
    gc.collect()
    free, total = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    target_gib = max(0, free_gib - safety_gib)
    chunks = []
    allocated = 0.0
    chunk_elems = int(chunk_gib * 1024**3 // 2)
    try:
        while allocated + chunk_gib <= target_gib:
            x = torch.empty((chunk_elems,), device="cuda", dtype=torch.float16)
            x.fill_(1)
            chunks.append(x)
            allocated += chunk_gib
    except RuntimeError as exc:
        err = str(exc).splitlines()[0]
    else:
        err = ""
    result = {
        "kind": "memory_capacity_probe",
        "total_gib": round(total / 1024**3, 3),
        "free_before_gib": round(free_gib, 3),
        "allocated_touched_gib": round(allocated, 3),
        "safety_gib": safety_gib,
        "error": err,
    }
    del chunks
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    torch.cuda.reset_peak_memory_stats()
    p = torch.cuda.get_device_properties(0)
    print(json.dumps({
        "kind": "device",
        "name": p.name,
        "capability": f"{p.major}.{p.minor}",
        "total_memory_gib": round(p.total_memory / 1024**3, 3),
        "sms": p.multi_processor_count,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }, ensure_ascii=False), flush=True)

    for n in [2048, 4096, 8192]:
        torch.cuda.reset_peak_memory_stats()
        print(json.dumps(bench_matmul(n, torch.float32, allow_tf32=False, reps=8), ensure_ascii=False), flush=True)
    for n in [2048, 4096, 8192]:
        torch.cuda.reset_peak_memory_stats()
        print(json.dumps(bench_matmul(n, torch.float32, allow_tf32=True, reps=8), ensure_ascii=False), flush=True)
    for dtype in [torch.float16, torch.bfloat16]:
        for n in [4096, 8192, 12288]:
            torch.cuda.reset_peak_memory_stats()
            print(json.dumps(bench_matmul(n, dtype, allow_tf32=True, reps=8), ensure_ascii=False), flush=True)

    torch.cuda.reset_peak_memory_stats()
    print(json.dumps(bench_copy(gib=2, reps=20), ensure_ascii=False), flush=True)
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps(bench_memory_capacity(chunk_gib=1.0, safety_gib=2.0), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
