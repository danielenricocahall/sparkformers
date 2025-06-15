import torch
from torch import nn
from sparkformers.utils.torch_utils import subtract_params, divide_by, get_param_diff


def test_subtract_params_basic():
    before = {
        "weight": torch.tensor([2.0, 4.0]),
        "bias": torch.tensor([1.0]),
    }
    after = {
        "weight": torch.tensor([1.0, 1.0]),
        "bias": torch.tensor([0.5]),
    }
    expected = {
        "weight": torch.tensor([1.0, 3.0]),
        "bias": torch.tensor([0.5]),
    }
    result = subtract_params(before, after)
    for key in expected:
        assert torch.allclose(result[key], expected[key])


def test_subtract_params_ignores_non_tensors():
    before = {"tensor": torch.tensor([5.0]), "meta": "not a tensor"}
    after = {"tensor": torch.tensor([2.0]), "meta": "still not a tensor"}
    result = subtract_params(before, after)
    assert "meta" not in result
    assert torch.allclose(result["tensor"], torch.tensor([3.0]))


def test_divide_by_scalar():
    params = {"a": torch.tensor([10.0, 20.0]), "b": torch.tensor([4.0])}
    expected = {"a": torch.tensor([5.0, 10.0]), "b": torch.tensor([2.0])}
    result = divide_by(params, 2)
    for key in expected:
        assert torch.allclose(result[key], expected[key])


def test_divide_by_ignores_non_tensors():
    params = {"tensor": torch.tensor([6.0]), "extra": "non-tensor"}
    result = divide_by(params, 3)
    assert "extra" not in result
    assert torch.allclose(result["tensor"], torch.tensor([2.0]))


def test_get_param_diff_returns_detached_clone():
    model = nn.Linear(2, 1)
    diff = get_param_diff(model)

    for k, v in model.state_dict().items():
        assert k in diff
        assert torch.equal(v.cpu(), diff[k])
        assert not diff[k].requires_grad
        assert diff[k] is not v  # Ensure it's a clone
