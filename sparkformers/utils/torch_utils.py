import torch


def subtract_params(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Subtract two PyTorch state_dicts: delta = before - after
    """
    return {
        key: before[key] - after[key]
        for key in before
        if key in after and isinstance(before[key], torch.Tensor)
    }


def add_params(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Add two PyTorch state_dicts: delta = before + after
    """
    return {
        key: before[key] + after[key]
        for key in before
        if key in after and isinstance(before[key], torch.Tensor)
    }


def divide_by(
    param_dict: dict[str, torch.Tensor], scalar: float
) -> dict[str, torch.Tensor]:
    """
    Divide all parameters in a PyTorch state_dict by a scalar
    """
    return {
        key: value / scalar
        for key, value in param_dict.items()
        if isinstance(value, torch.Tensor)
    }
