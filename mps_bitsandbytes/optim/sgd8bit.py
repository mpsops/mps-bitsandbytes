"""
8-bit SGD optimizer with momentum for MPS
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable

from .adam8bit import quantize_state, dequantize_state


class SGD8bit(Optimizer):
    """
    8-bit SGD optimizer with momentum.

    Stores momentum buffer in int8 format for memory efficiency.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (required)
        momentum: Momentum factor (default: 0)
        dampening: Dampening for momentum (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        nesterov: Enables Nesterov momentum (default: False)
        block_size: Block size for quantization (default: 256)
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        block_size: int = 256,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and zero dampening")

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                       weight_decay=weight_decay, nesterov=nesterov,
                       block_size=block_size)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            block_size = group['block_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SGD8bit does not support sparse gradients")

                # Weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Momentum
                if momentum != 0:
                    state = self.state[p]

                    if len(state) == 0:
                        state['momentum_int8'], state['momentum_absmax'] = quantize_state(
                            torch.zeros_like(p, memory_format=torch.preserve_format),
                            block_size
                        )

                    buf = dequantize_state(
                        state['momentum_int8'], state['momentum_absmax'],
                        block_size, torch.float32
                    )
                    grad_fp32 = grad.float()
                    buf.mul_(momentum).add_(grad_fp32, alpha=1 - dampening)

                    if nesterov:
                        grad_fp32 = grad_fp32.add(buf, alpha=momentum)
                    else:
                        grad_fp32 = buf

                    # Requantize
                    state['momentum_int8'], state['momentum_absmax'] = quantize_state(buf, block_size)
                    grad = grad_fp32.to(p.dtype)

                p.add_(grad, alpha=-group['lr'])

        return loss
