import time
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
print("GPU:", torch.cuda.get_device_name(0) if device=="cuda" else "None")

x = torch.randn(6000, 6000, device=device)
torch.cuda.synchronize() if device=="cuda" else None
t0 = time.time()
y = x @ x
torch.cuda.synchronize() if device=="cuda" else None
print("Matmul time (s):", round(time.time() - t0, 3))