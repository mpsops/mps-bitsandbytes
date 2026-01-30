"""
8-bit Lion optimizer for MPS

Lion (Evolved Sign Momentum) uses only sign operations, making it
naturally suited for 8-bit quantization with minimal accuracy loss.
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, Optional, Callable

from .adam8bit import quantize_state, dequantize_state


class Lion8bit(Optimizer):
    """
    8-bit Lion optimizer with blockwise quantization.

    Lion uses sign-based updates which are robust to quantization.
    Only stores one momentum state (vs two for Adam), so memory
    savings are even more significant.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-4)
        betas: Coefficients for computing running averages (default: (0.9, 0.99))
        weight_decay: Weight decay coefficient (default: 0)
        block_size: Block size for quantization (default: 256)

    Reference:
        "Symbolic Discovery of Optimization Algorithms"
        https://arxiv.org/abs/2302.06675
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0,
        block_size: int = 256,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
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
            beta1, beta2 = group['betas']
            block_size = group['block_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion8bit does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg_int8'], state['exp_avg_absmax'] = quantize_state(
                        torch.zeros_like(p, memory_format=torch.preserve_format),
                        block_size
                    )

                # Dequantize momentum (FP32 for stability)
                exp_avg = dequantize_state(
                    state['exp_avg_int8'], state['exp_avg_absmax'],
                    block_size, torch.float32
                )

                # Weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Lion update: use sign of interpolation
                grad_fp32 = grad.float()
                update = exp_avg.mul(beta1).add(grad_fp32, alpha=1 - beta1)
                p.add_(update.sign().to(p.dtype), alpha=-group['lr'])

                # Update momentum for next step
                exp_avg.mul_(beta2).add_(grad_fp32, alpha=1 - beta2)

                # Requantize
                state['exp_avg_int8'], state['exp_avg_absmax'] = quantize_state(exp_avg, block_size)

        return loss
