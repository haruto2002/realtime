import argparse
import time

import torch
import torch.nn as nn


class SmallConvNet(nn.Module):
    def __init__(self, channels: int = 16, depth: int = 2):
        super().__init__()
        layers = []
        in_ch = 3
        for _ in range(depth):
            layers += [
                nn.Conv2d(in_ch, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = channels
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def run_sequential(model1, model2, x1, x2, n_iter: int):
    sync()
    t0 = time.perf_counter()

    for _ in range(n_iter):
        y1 = model1(x1)
        y2 = model2(x2)

    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter


@torch.no_grad()
def run_parallel_sync_each(model1, model2, x1, x2, stream1, stream2, n_iter: int):
    sync()
    t0 = time.perf_counter()

    for _ in range(n_iter):
        with torch.cuda.stream(stream1):
            y1 = model1(x1)

        with torch.cuda.stream(stream2):
            y2 = model2(x2)

        stream1.synchronize()
        stream2.synchronize()

    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter


@torch.no_grad()
def run_parallel_sync_last(model1, model2, x1, x2, stream1, stream2, n_iter: int):
    sync()
    t0 = time.perf_counter()

    for _ in range(n_iter):
        with torch.cuda.stream(stream1):
            y1 = model1(x1)

        with torch.cuda.stream(stream2):
            y2 = model2(x2)

    stream1.synchronize()
    stream2.synchronize()

    sync()
    t1 = time.perf_counter()
    return (t1 - t0) / n_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iter", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU is required"

    device = "cuda"
    dtype = torch.float16 if args.fp16 else torch.float32

    torch.backends.cudnn.benchmark = True

    model1 = SmallConvNet(args.channels, args.depth).to(device).eval()
    model2 = SmallConvNet(args.channels, args.depth).to(device).eval()

    if args.fp16:
        model1 = model1.half()
        model2 = model2.half()

    x1 = torch.randn(args.batch, 3, args.size, args.size, device=device, dtype=dtype)
    x2 = torch.randn(args.batch, 3, args.size, args.size, device=device, dtype=dtype)

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # Warmup
    run_sequential(model1, model2, x1, x2, args.warmup)
    run_parallel_sync_each(model1, model2, x1, x2, stream1, stream2, args.warmup)
    run_parallel_sync_last(model1, model2, x1, x2, stream1, stream2, args.warmup)

    seq = run_sequential(model1, model2, x1, x2, args.iter)
    # par_each = run_parallel_sync_each(
    #     model1, model2, x1, x2, stream1, stream2, args.iter
    # )
    # par_last = run_parallel_sync_last(
    #     model1, model2, x1, x2, stream1, stream2, args.iter
    # )

    print("==== CUDA Stream Parallel Test ====")
    print(f"GPU:        {torch.cuda.get_device_name()}")
    print(f"input:      batch={args.batch}, size={args.size}x{args.size}")
    print(f"model:      channels={args.channels}, depth={args.depth}")
    print(f"dtype:      {dtype}")
    print()
    print(f"sequential:          {seq * 1000:.4f} ms / iter")
    # print(f"parallel sync each:  {par_each * 1000:.4f} ms / iter")
    # print(f"parallel sync last:  {par_last * 1000:.4f} ms / iter")
    # print()
    # print(f"speedup sync each:   {seq / par_each:.3f}x")
    # print(f"speedup sync last:   {seq / par_last:.3f}x")


if __name__ == "__main__":
    main()
