
import torch.nn as nn

# This is our model, which is from value network of ELF. Compared with model in ELF, we lack the policy network. 

class Model81(nn.Module):
    # Base class for an RL model, it is a wrapper for ``nn.Module``'''
    def __init__(self):
        super().__init__()
