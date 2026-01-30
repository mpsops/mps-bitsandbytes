"""
Paged optimizers for MPS

Offload optimizer states to CPU memory when not in use, enabling training
of larger models by trading compute for memory.
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, Optional, Callable, Dict, Any


class PagedAdamW(Optimizer):
    """
    Paged AdamW optimizer that offloads states to CPU.

    Optimizer states (m, v) are stored on CPU and moved to GPU only
    during the step. This allows training larger models at the cost
    of CPU-GPU transfer overhead.

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 1e-2)
        page_to_cpu: Whether to offload states to CPU (default: True)

    Example:
        >>> # Train a model larger than GPU memory
        >>> model = LargeModel().to('mps')
        >>> optimizer = PagedAdamW(model.parameters(), lr=1e-4)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        page_to_cpu: bool = True,
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
                       page_to_cpu=page_to_cpu)
        super().__init__(params, defaults)

    def _page_in(self, state: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move optimizer states from CPU to device."""
        exp_avg = state['exp_avg'].to(device, non_blocking=True)
        exp_avg_sq = state['exp_avg_sq'].to(device, non_blocking=True)
        return exp_avg, exp_avg_sq

    def _page_out(self, state: Dict[str, Any], exp_avg: torch.Tensor, exp_avg_sq: torch.Tensor):
        """Move optimizer states back to CPU."""
        state['exp_avg'] = exp_avg.to('cpu', non_blocking=True)
        state['exp_avg_sq'] = exp_avg_sq.to('cpu', non_blocking=True)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            page_to_cpu = group['page_to_cpu']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PagedAdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Initialize on CPU if paging enabled
                    storage_device = 'cpu' if page_to_cpu else p.device
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=storage_device
                    )
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=storage_device
                    )

                state['step'] += 1

                # Page in states
                if page_to_cpu:
                    exp_avg, exp_avg_sq = self._page_in(state, p.device)
                    torch.mps.synchronize() if p.device.type == 'mps' else None
                else:
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute step
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Page out states
                if page_to_cpu:
                    self._page_out(state, exp_avg, exp_avg_sq)

        return loss


class PagedAdam(PagedAdamW):
    """
    Paged Adam optimizer (L2 weight decay, not decoupled).

    Same as PagedAdamW but with L2 regularization applied to gradients
    instead of decoupled weight decay.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        page_to_cpu: bool = True,
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, page_to_cpu=page_to_cpu)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step with L2 weight decay."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            page_to_cpu = group['page_to_cpu']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PagedAdam does not support sparse gradients")

                # L2 weight decay (applied to grad, not weights)
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    storage_device = 'cpu' if page_to_cpu else p.device
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=storage_device
                    )
                    state['exp_avg_sq'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=storage_device
                    )

                state['step'] += 1

                if page_to_cpu:
                    exp_avg, exp_avg_sq = self._page_in(state, p.device)
                    torch.mps.synchronize() if p.device.type == 'mps' else None
                else:
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

                if page_to_cpu:
                    self._page_out(state, exp_avg, exp_avg_sq)

        return loss


class PagedLion(Optimizer):
    """
    Paged Lion optimizer that offloads momentum to CPU.

    Lion only has one momentum state, making paging even more efficient.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0,
        page_to_cpu: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay,
                       page_to_cpu=page_to_cpu)
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
            page_to_cpu = group['page_to_cpu']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    storage_device = 'cpu' if page_to_cpu else p.device
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=storage_device
                    )

                # Page in
                if page_to_cpu:
                    exp_avg = state['exp_avg'].to(p.device, non_blocking=True)
                    torch.mps.synchronize() if p.device.type == 'mps' else None
                else:
                    exp_avg = state['exp_avg']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Lion update
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-group['lr'])

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Page out
                if page_to_cpu:
                    state['exp_avg'] = exp_avg.to('cpu', non_blocking=True)

        return loss
