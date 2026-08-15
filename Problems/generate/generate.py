
import torch

def generate_coords(num_samples, size):
    # --- Generate random coords                
    return torch.rand(num_samples, size, 2)


def generate_demand_as_kool(num_samples, size):
    return torch.randint(low=1, high=9, size = (num_samples, size))