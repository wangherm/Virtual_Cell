"""New Round15 heads; historical Round14 initializations remain unchanged."""
import torch

from .round14 import ResponseHead


class FixedIDHead(ResponseHead):
    """Repair the zero-input/zero-ID/zero-output dead branch only when requested."""
    def __init__(self, target_dim, background_dim, n_ids, rank, arm_spec, seed):
        super().__init__(target_dim, background_dim, n_ids, rank, arm_spec)
        if arm_spec.get('fix_id', False):
            generator = torch.Generator(device='cpu').manual_seed(seed + 15002)
            with torch.no_grad():
                self.ids.weight[1:].copy_(.01 * torch.randn(n_ids-1, arm_spec['width'], generator=generator))
                self.ids.weight[0].zero_()
