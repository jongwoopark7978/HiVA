import sys, time, torch

secs = int(sys.argv[2]) if len(sys.argv) > 2 else 60

n = torch.cuda.device_count()

if n == 0:
    raise RuntimeError("No CUDA devices are visible.")

buf = []
for i in range(n):
    x = torch.randn(1024, 1024, device=f"cuda:{i}")
    y = torch.randn(1024, 1024, device=f"cuda:{i}")
    buf.append((i, x, y))
    print(f"gpu {i}: {torch.cuda.get_device_name(i)} ready", flush=True)

# end = time.time() + secs
# while time.time() < end:
while True:
    for i, x, y in buf:
        _ = (x @ y).mean()
    for i, _, _ in buf:
        torch.cuda.synchronize(i)
    time.sleep(1)

print("done")