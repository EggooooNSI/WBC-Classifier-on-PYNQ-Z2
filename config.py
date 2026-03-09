import torch

gpu_id = 1
gpu_name = 'cuda:{}'.format(gpu_id)
device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")