"""New Round15 heads; historical Round14 initializations remain unchanged."""
import torch
from torch import nn
import torch.nn.functional as F

from .round14 import ResponseHead, BASE


_ARMS = {
    'background_only': {'representation':'none'},
    'id_background_fixed': {'representation':'none','use_id':True,'fix_id':True},
    'local_additive_noid': {},
    'local_factor_noid': {'interaction':True},
    'local_additive_id': {'use_id':True},
    'local_factor_id': {'interaction':True,'use_id':True},
    **{f'shuffle{j}_factor_noid':{'representation':f'shuffled_{j}','interaction':True} for j in (1,2,3)},
    'gaussian_factor_noid': {'representation':'random','interaction':True},
    'coverage_only': {'representation':'coverage'},
    'local_target_only_matched': {'background':'zero'},
    'shuffle_target_only_matched': {'background':'zero','representation':'shuffled_1'},
    'ridge_residual': {'reference':'ridge','interaction':True},
    'reference_residual': {'reference':'reference','interaction':True},
    'neighbor_response_residual': {'reference':'neighbor','interaction':True},
    'film_small': {'architecture':'film'},
    'shared_centered_residual': {'reference':'shared','interaction':True},
    'no_context_only_output': {'architecture':'no_context','interaction':True},
    'amplitude_direction': {'architecture':'amplitude','interaction':True},
    'control_pca_factor': {'background':'control_pca','interaction':True},
    'control_moments_factor': {'background':'control_moments','interaction':True},
    'factor_response_rank64': {'rank':64,'interaction':True},
    'factor_response_rank16': {'rank':16,'interaction':True},
}
ARMS = {k:{**BASE,'architecture':'standard','reference':None,'rank':32,'fix_id':False,**v} for k,v in _ARMS.items()}
REUSE = {'background_only':'background_only','local_additive_noid':'local_additive',
         'local_factor_noid':'local_factor','local_additive_id':'local_additive_id',
         'local_factor_id':'local_factor_id','local_target_only_matched':'target_only'}
VARIANTS = ('mse_reference','huber','mse_cosine','mse_target_contrastive','gene_scale_robust',
            'measurement_precision','replicate_consistency','replicate_balanced_sampling',
            'lr_1e4','lr_1e3','wd_1e3','wd_1e1')
BLOCKED = {
    'control_moments_factor':'Real control-cell variances and row/control lineage are not present in prepared pseudobulks.',
    'measurement_precision':'Requires measured within-group variance or biological replicates; n_cells is not precision.',
    'replicate_consistency':'Requires independent matched-control resamples with original cell IDs and audited split membership.',
    'replicate_balanced_sampling':'Existing batch-balanced rows do not establish biological replicate identity; equivalence cannot be assumed.',
}


class FixedIDHead(ResponseHead):
    """Repair the zero-input/zero-ID/zero-output dead branch only when requested."""
    def __init__(self, target_dim, background_dim, n_ids, rank, arm_spec, seed):
        super().__init__(target_dim, background_dim, n_ids, rank, arm_spec)
        if arm_spec.get('fix_id', False):
            generator = torch.Generator(device='cpu').manual_seed(seed + 15002)
            with torch.no_grad():
                self.ids.weight[1:].copy_(.01 * torch.randn(n_ids-1, arm_spec['width'], generator=generator))
                self.ids.weight[0].zero_()


class ExperimentalHead(FixedIDHead):
    def __init__(self, target_dim, background_dim, n_ids, rank, arm_spec, seed):
        super().__init__(target_dim,background_dim,n_ids,rank,arm_spec,seed)
        self.architecture = arm_spec['architecture']; self.reference = arm_spec['reference']
        width = arm_spec['width']
        if self.reference: self.reference_input = nn.Linear(rank+1,width,bias=False)
        if self.architecture == 'film':
            self.modulation = nn.Linear(width,2*width)
            nn.init.zeros_(self.modulation.weight); nn.init.zeros_(self.modulation.bias)
        if self.architecture == 'no_context': self.context_out.weight.requires_grad_(False)
        if self.architecture == 'amplitude':
            self.amplitude = nn.Linear(width,1)
            nn.init.zeros_(self.amplitude.weight); nn.init.constant_(self.amplitude.bias,-4.)
            # A positive small amplitude and nonzero direction avoid another dead product.
            generator = torch.Generator().manual_seed(seed+15200)
            with torch.no_grad():
                self.target_out.weight.copy_(.01*torch.randn(self.target_out.weight.shape,generator=generator))
                self.context_out.weight.copy_(.01*torch.randn(self.context_out.weight.shape,generator=generator))

    def forward(self,target,background,ids,reference=None):
        hp = self.target(target)
        if self.use_id: hp = hp+self.ids(ids)
        if self.reference:
            if reference is None: raise ValueError('Reference input required')
            hp = hp+self.reference_input(reference)
        hp = hp.tanh(); hc = self.context(background).tanh()
        if self.architecture == 'film':
            scale,shift = self.modulation(hc).chunk(2,dim=-1)
            hp = hp*(1+.1*scale.tanh())+.1*shift.tanh()
        # Preserve historical floating-point operation order for exact reuse.
        if self.architecture == 'no_context': out = self.target_out(hp)+self.bias
        else: out = self.target_out(hp)+self.context_out(hc)+self.bias
        if self.interaction: out = out+self.cross_out(hp*hc)
        if self.architecture == 'amplitude':
            out = F.softplus(self.amplitude(hp+hc))*F.normalize(out,dim=-1,eps=1e-8)
        return out


def contrastive_loss(prediction, truth, groups, targets, minimum_norm):
    """Supervised response retrieval within identical measured conditions.

    Same-target observations are positives (sampling duplicates are not new replicates).
    Other-target responses with fit cosine >=.9 are neutral rather than hard negatives.
    """
    p = F.normalize(prediction,dim=-1,eps=1e-8); y = F.normalize(truth,dim=-1,eps=1e-8)
    valid = truth.norm(dim=-1)>minimum_norm
    same = (groups[:,None]==groups[None,:]) & valid[:,None] & valid[None,:]
    positive = same & (targets[:,None]==targets[None,:])
    negative = same & (targets[:,None]!=targets[None,:]) & ((y@y.T)<.9)
    usable = positive.any(1)&negative.any(1)
    if not usable.any(): return prediction.sum()*0, 0
    logits = ((p@y.T)/.1)[usable]
    numerator = torch.logsumexp(logits.masked_fill(~positive[usable],-torch.inf),dim=1)
    denominator = torch.logsumexp(logits.masked_fill(~(positive|negative)[usable],-torch.inf),dim=1)
    return (denominator-numerator).mean(), int(usable.sum())
