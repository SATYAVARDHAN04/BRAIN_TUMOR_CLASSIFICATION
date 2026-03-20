import torch

if torch.cuda.is_available():
    print(f"✅ GPU Found: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("❌ No NVIDIA GPU found (or CUDA is not installed).")