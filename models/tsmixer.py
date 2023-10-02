import torch
import torch.nn as nn

from models.common import RevIN
from models.common import ResBlock


class TSMixerRevIN(nn.Module):
    """Implementation of TSMixerRevIN."""

    def __init__(self, input_shape, pred_len, n_block, dropout, ff_dim, target_slice):
        super(TSMixerRevIN, self).__init__()
        
        self.target_slice = target_slice
        
        self.rev_norm = RevIN(input_shape[-1])
        
        self.res_blocks = nn.ModuleList([ResBlock(input_shape, dropout, ff_dim) for _ in range(n_block)])
        
        self.linear = nn.Linear(input_shape[0], pred_len)
        
    def forward(self, x):
        
        x = self.rev_norm(x, 'norm')
        
        for res_block in self.res_blocks:
            x = res_block(x)

        if self.target_slice:
            x = x[:, :, self.target_slice]

        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)
        
        x = self.rev_norm(x, 'denorm', self.target_slice)
        
        return x