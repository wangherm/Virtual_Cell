import torch

from vcell.round14 import ResponseHead, ARMS
from vcell.round15_heads import FixedIDHead


def test_fixed_id_gradients_fit_and_unknown_padding():
    spec = {**ARMS['id_background'], 'fix_id': True}
    torch.manual_seed(4)
    model = FixedIDHead(3, 2, 4, 2, spec, 17)
    x = torch.zeros(9, 3); b = torch.zeros(9, 2); ids = torch.tensor([1,2,3]*3)
    y = torch.tensor([[1.,0.],[-1.,1.],[0.,-1.]]*3)
    optimizer = torch.optim.Adam(model.parameters(), lr=.05)
    losses = []
    for step in range(100):
        optimizer.zero_grad(); loss = (model(x,b,ids)-y).square().mean(); loss.backward()
        if step == 0: assert model.target_out.weight.grad.abs().sum() > 0
        if step == 1: assert model.ids.weight.grad[1:].abs().sum() > 0
        assert model.ids.weight.grad[0].abs().sum() == 0
        optimizer.step(); losses.append(loss.item())
    assert losses[-1] < losses[0] * .05
    assert not model.ids.weight[0].any()


def test_repair_preserves_other_parameters_and_global_rng():
    spec = ARMS['id_background']
    torch.manual_seed(4); old = ResponseHead(3,2,5,2,spec); old_rng = torch.get_rng_state()
    torch.manual_seed(4); fixed = FixedIDHead(3,2,5,2,{**spec,'fix_id':True},17)
    assert torch.equal(old_rng, torch.get_rng_state())
    for name, value in old.state_dict().items():
        if name != 'ids.weight': torch.testing.assert_close(value, fixed.state_dict()[name], rtol=0, atol=0)
    torch.manual_seed(4); untouched = FixedIDHead(3,2,5,2,spec,17)
    for name, value in old.state_dict().items(): torch.testing.assert_close(value, untouched.state_dict()[name], rtol=0, atol=0)
