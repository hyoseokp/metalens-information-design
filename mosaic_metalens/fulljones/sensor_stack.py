from __future__ import annotations

from collections.abc import Callable

import torch


def _broadcast_transfer(transfer: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if transfer.ndim == 2:
        return transfer.unsqueeze(0).unsqueeze(0)
    if transfer.ndim == 3:
        return transfer.unsqueeze(0)
    if transfer.ndim == 4:
        return transfer
    raise ValueError("transfer tensor must have 2, 3, or 4 dimensions.")


def apply_sensor_stack(
    sensor_field: torch.Tensor,
    microlens_transfer: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
    stack_transfer: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    field = sensor_field
    for transfer in (microlens_transfer, stack_transfer):
        if transfer is None:
            continue
        if callable(transfer):
            field = transfer(field)
        else:
            field = field * _broadcast_transfer(transfer.to(field.device), field)
    return field


def photodiode_irradiance(field: torch.Tensor) -> torch.Tensor:
    return field.abs().square()
