import torch

print("PyTorch Version:", torch.__version__)
print("ROCm (HIP) Version:", torch.version.hip)
print("CUDA available (ROCm uses this API):", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))