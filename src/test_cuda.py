import torch
print(torch.version.cuda)      # CUDA version PyTorch was built with
print(torch.cuda.is_available())
print(torch.cuda.device_count())
