import torch

print("PyTorch version:", torch.__version__)
print("CUDA runtime bundled with PyTorch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GB):", round(
        torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
    ))

    # Execute a real tensor operation on the GPU
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("GPU tensor test: PASSED")
    print("Output device:", y.device)
else:
    print("GPU not available. We will troubleshoot before training.")
