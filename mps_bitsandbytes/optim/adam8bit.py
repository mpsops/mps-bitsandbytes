"""
8-bit Adam and AdamW optimizers for MPS

Stores optimizer states (m, v) in 8-bit, reducing optimizer memory by ~75%.
Uses dynamic quantization with blockwise scaling for accuracy.
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, Optional, Callable


def quantize_state(state: torch.Tensor, block_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize optimizer state to int8 with blockwise scaling."""
    orig_shape = state.shape
    state_flat = state.flatten().float()

    # Pad to block_size multiple
    numel = state_flat.numel()
    padded_numel = ((numel + block_size - 1) // block_size) * block_size
    if padded_numel > numel:
        state_flat = torch.nn.functional.pad(state_flat, (0, padded_numel - numel))

    # Reshape to blocks
    state_blocks = state_flat.view(-1, block_size)

    # Compute absmax per block
    absmax = state_blocks.abs().max(dim=1).values.clamp(min=1e-8)

    # Quantize to int8
    state_normalized = state_blocks / absmax.unsqueeze(1)
    state_int8 = (state_normalized * 127).round().clamp(-127, 127).to(torch.int8)

    return state_int8.flatten()[:numel].view(orig_shape), absmax


def dequantize_state(state_int8: torch.Tensor, absmax: torch.Tensor,
                     block_size: int = 256, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dequantize int8 optimizer state back to float."""
    orig_shape = state_int8.shape
    state_flat = state_int8.flatten().float()

    # Pad to block_size multiple
    numel = state_flat.numel()
    padded_numel = ((numel + block_size - 1) // block_size) * block_size
    if padded_numel > numel:
        state_flat = torch.nn.functional.pad(state_flat, (0, padded_numel - numel))

    # Reshape and dequantize
    state_blocks = state_flat.view(-1, block_size)
    state_dequant = (state_blocks / 127.0) * absmax.unsqueeze(1)

    return state_dequant.flatten()[:numel].view(orig_shape).to(dtype)


def quantize_state_unsigned(state: torch.Tensor, block_size: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize non-negative state (like exp_avg_sq) to uint8 with dynamic exponent.

    Uses log-linear quantization to preserve small values that matter for Adam's
    denominator. Stores (max_val, min_nonzero) per block to reconstruct values.
    """
    orig_shape = state.shape
    state_flat = state.flatten().float().clamp(min=0)

    # Pad to block_size multiple
    numel = state_flat.numel()
    padded_numel = ((numel + block_size - 1) // block_size) * block_size
    if padded_numel > numel:
        state_flat = torch.nn.functional.pad(state_flat, (0, padded_numel - numel))

    # Reshape to blocks
    state_blocks = state_flat.view(-1, block_size)

    # Per-block: store max value and use sqrt scaling to compress dynamic range
    # sqrt(x) compresses [0, 1e-4] to [0, 0.01] - much better for quantization
    block_max = state_blocks.max(dim=1).values.clamp(min=1e-12)

    # Normalize and take sqrt to compress dynamic range
    state_normalized = state_blocks / block_max.unsqueeze(1)
    state_sqrt = state_normalized.sqrt()  # [0,1] -> [0,1] but small values become larger

    # Quantize sqrt values to uint8
    state_uint8 = (state_sqrt * 255).round().clamp(0, 255).to(torch.uint8)

    return state_uint8.flatten()[:numel].view(orig_shape), block_max


def dequantize_state_unsigned(state_uint8: torch.Tensor, block_max: torch.Tensor,
                              block_size: int = 256, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Dequantize uint8 state back to float, reversing sqrt compression."""
    orig_shape = state_uint8.shape
    state_flat = state_uint8.flatten().float()

    # Pad to block_size multiple
    numel = state_flat.numel()
    padded_numel = ((numel + block_size - 1) // block_size) * block_size
    if padded_numel > numel:
        state_flat = torch.nn.functional.pad(state_flat, (0, padded_numel - numel))

    # Reshape and dequantize
    state_blocks = state_flat.view(-1, block_size)

    # Reverse: uint8 -> [0,1] sqrt -> square -> scale by max
    state_sqrt = state_blocks / 255.0
    state_normalized = state_sqrt * state_sqrt  # square to reverse sqrt
    state_dequant = state_normalized * block_max.unsqueeze(1)

    return state_dequant.flatten()[:numel].view(orig_shape).to(dtype)


class Adam8bit(Optimizer):
    """
    8-bit Adam optimizer with blockwise quantization.

    Stores first moment (m) and second moment (v) in int8 format,
    reducing optimizer memory by ~75% compared to standard Adam.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        block_size: Block size for quantization (default: 256)

    Example:
        >>> model = MyModel().to('mps')
        >>> optimizer = Adam8bit(model.parameters(), lr=1e-4)
        >>> for data, target in dataloader:
        ...     optimizer.zero_grad()
        ...     loss = criterion(model(data), target)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        block_size: int = 256,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
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
                    raise RuntimeError("Adam8bit does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Initialize in 8-bit (exp_avg uses signed, exp_avg_sq uses unsigned)
                    state['exp_avg_int8'], state['exp_avg_absmax'] = quantize_state(
                        torch.zeros_like(p, memory_format=torch.preserve_format),
                        block_size
                    )
                    state['exp_avg_sq_uint8'], state['exp_avg_sq_max'] = quantize_state_unsigned(
                        torch.zeros_like(p, memory_format=torch.preserve_format),
                        block_size
                    )

                state['step'] += 1

                # Dequantize states (always in FP32 for numerical stability)
                exp_avg = dequantize_state(
                    state['exp_avg_int8'], state['exp_avg_absmax'],
                    block_size, torch.float32
                )
                exp_avg_sq = dequantize_state_unsigned(
                    state['exp_avg_sq_uint8'], state['exp_avg_sq_max'],
                    block_size, torch.float32
                )

                # Weight decay
                grad_fp32 = grad.float()
                if group['weight_decay'] != 0:
                    grad_fp32 = grad_fp32.add(p.float(), alpha=group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute step (in FP32 for numerical stability)
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                update = exp_avg / denom * (-step_size)
                p.add_(update.to(p.dtype))

                # Requantize states
                state['exp_avg_int8'], state['exp_avg_absmax'] = quantize_state(exp_avg, block_size)
                state['exp_avg_sq_uint8'], state['exp_avg_sq_max'] = quantize_state_unsigned(exp_avg_sq, block_size)

        return loss


class AdamW8bit(Optimizer):
    """
    8-bit AdamW optimizer with decoupled weight decay.

    Like Adam8bit but with decoupled weight decay regularization
    as described in "Decoupled Weight Decay Regularization".

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 1e-2)
        block_size: Block size for quantization (default: 256)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        block_size: int = 256,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
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
                    raise RuntimeError("AdamW8bit does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_int8'], state['exp_avg_absmax'] = quantize_state(
                        torch.zeros_like(p, memory_format=torch.preserve_format),
                        block_size
                    )
                    state['exp_avg_sq_uint8'], state['exp_avg_sq_max'] = quantize_state_unsigned(
                        torch.zeros_like(p, memory_format=torch.preserve_format),
                        block_size
                    )

                state['step'] += 1

                # Dequantize states (always in FP32 for numerical stability)
                exp_avg = dequantize_state(
                    state['exp_avg_int8'], state['exp_avg_absmax'],
                    block_size, torch.float32
                )
                exp_avg_sq = dequantize_state_unsigned(
                    state['exp_avg_sq_uint8'], state['exp_avg_sq_max'],
                    block_size, torch.float32
                )

                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update biased first moment estimate
                grad_fp32 = grad.float()
                exp_avg.mul_(beta1).add_(grad_fp32, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute step (in FP32 for numerical stability)
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                update = exp_avg / denom * (-step_size)
                p.add_(update.to(p.dtype))

                # Requantize states
                state['exp_avg_int8'], state['exp_avg_absmax'] = quantize_state(exp_avg, block_size)
                state['exp_avg_sq_uint8'], state['exp_avg_sq_max'] = quantize_state_unsigned(exp_avg_sq, block_size)

        return loss
