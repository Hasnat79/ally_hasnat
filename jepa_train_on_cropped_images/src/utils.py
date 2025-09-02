import torch

def get_device_configuration():
    # Device configuration
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f'Using cuda and device: "{torch.cuda.get_device_name(0)}"')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        print("Using mps")
    else:
        DEVICE = torch.device("cpu")
        print("Using cpu")
    return DEVICE
