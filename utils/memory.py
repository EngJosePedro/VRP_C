import torch

def mem(tag=""):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    m = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[{tag}] alloc={a:.3f} GB | reserved={r:.3f} GB | max={m:.3f} GB")

def tensor_mem(t, name):
    mb = t.numel() * t.element_size() / 1024**2
    print(f"{name}: {mb:.3f} MB")