import torch.nn as nn


def clone(model: nn.Module) -> nn.Module:
    copy = model.__class__(model.config)
    copy.load_state_dict(model.state_dict())
    return copy


def get_mergeable_variables(model: nn.Module) -> list[nn.Parameter]:
    for name, child in model.named_children():
        if name in ["classifier", "dropout"]:
            continue
        else:
            body = child
    return [p for p in body.parameters() if p.requires_grad]


def set_mergeable_variables(model: nn.Module, new_params: list[nn.Parameter]):
    params = get_mergeable_variables(model)
    assert len(params) == len(new_params), "Parameter list length mismatch."

    for p, new_p in zip(params, new_params):
        p.data.copy_(new_p.data if hasattr(new_p, "data") else new_p)
