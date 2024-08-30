import torch
# check if gpu is available
print(torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())
print(torch.cuda.get_device_name(0))