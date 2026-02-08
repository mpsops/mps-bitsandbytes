"""
Paged optimizers for MPS

Offload optimizer states to CPU memory when not in use, enabling training
of larger models by trading compute for memory.
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, Optional, Callable, Dict, Any, List


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
        self._pending_sync = False

    def synchronize(self):
        """Ensure all async transfers are complete. Call before accessing state."""
        if self._pending_sync:
            torch.mps.synchronize()
            self._pending_sync = False

    def __del__(self):
        """Ensure sync on cleanup to prevent crashes."""
        if hasattr(self, '_pending_sync') and self._pending_sync:
            try:
                torch.mps.synchronize()
            except Exception:
                pass  # Ignore errors during cleanup

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step with async prefetching."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Sync from previous step's async page-out (lazy sync)
        if self._pending_sync:
            torch.mps.synchronize()
            self._pending_sync = False

        has_mps_params = False

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            page_to_cpu = group['page_to_cpu']

            # Collect params needing steps for prefetch optimization
            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if not params_with_grad:
                continue

            # Batch small params (numel < 32768) for efficient transfer
            small_params: List[torch.Tensor] = []
            large_params: List[torch.Tensor] = []
            for p in params_with_grad:
                if p.numel() < 32768:
                    small_params.append(p)
                else:
                    large_params.append(p)

            # Prefetch first large param's states
            prefetched_states: Optional[Dict[str, Any]] = None
            if page_to_cpu and large_params:
                state = self.state[large_params[0]]
                if len(state) > 0:
                    prefetched_states = {
                        'exp_avg': state['exp_avg'].to(large_params[0].device, non_blocking=True),
                        'exp_avg_sq': state['exp_avg_sq'].to(large_params[0].device, non_blocking=True),
                        'param': large_params[0],
                    }

            # Process large params with prefetch overlap
            for i, p in enumerate(large_params):
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PagedAdamW does not support sparse gradients")

                has_mps_params = has_mps_params or (p.device.type == 'mps')
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    storage_device = 'cpu' if page_to_cpu else p.device
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, device=storage_device)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, device=storage_device)

                state['step'] += 1

                # Use prefetched states if available for this param
                if prefetched_states is not None and prefetched_states['param'] is p:
                    exp_avg = prefetched_states['exp_avg']
                    exp_avg_sq = prefetched_states['exp_avg_sq']
                elif page_to_cpu:
                    exp_avg = state['exp_avg'].to(p.device)
                    exp_avg_sq = state['exp_avg_sq'].to(p.device)
                else:
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                # Start prefetching NEXT param's states (overlap with compute)
                prefetched_states = None
                if page_to_cpu and i + 1 < len(large_params):
                    next_p = large_params[i + 1]
                    next_state = self.state[next_p]
                    if len(next_state) > 0:
                        prefetched_states = {
                            'exp_avg': next_state['exp_avg'].to(next_p.device, non_blocking=True),
                            'exp_avg_sq': next_state['exp_avg_sq'].to(next_p.device, non_blocking=True),
                            'param': next_p,
                        }

                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

                if page_to_cpu:
                    state['exp_avg'] = exp_avg.to('cpu', non_blocking=True)
                    state['exp_avg_sq'] = exp_avg_sq.to('cpu', non_blocking=True)

            # Process small params (simple loop, no prefetch overhead)
            for p in small_params:
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("PagedAdamW does not support sparse gradients")

                has_mps_params = has_mps_params or (p.device.type == 'mps')
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    storage_device = 'cpu' if page_to_cpu else p.device
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format, device=storage_device)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format, device=storage_device)

                state['step'] += 1

                if page_to_cpu:
                    exp_avg = state['exp_avg'].to(p.device)
                    exp_avg_sq = state['exp_avg_sq'].to(p.device)
                else:
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

                if page_to_cpu:
                    state['exp_avg'] = exp_avg.to('cpu', non_blocking=True)
                    state['exp_avg_sq'] = exp_avg_sq.to('cpu', non_blocking=True)

        # Mark that we need sync before next step
        if has_mps_params:
            self._pending_sync = True

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

        # Sync from previous step's async page-out
        if self._pending_sync:
            torch.mps.synchronize()
            self._pending_sync = False

        has_mps_params = False

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

                has_mps_params = has_mps_params or (p.device.type == 'mps')
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
                    exp_avg = state['exp_avg'].to(p.device)
                    exp_avg_sq = state['exp_avg_sq'].to(p.device)
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
                    state['exp_avg'] = exp_avg.to('cpu', non_blocking=True)
                    state['exp_avg_sq'] = exp_avg_sq.to('cpu', non_blocking=True)

        if has_mps_params:
            self._pending_sync = True

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
        self._pending_sync = False

    def synchronize(self):
        """Ensure all async transfers are complete. Call before accessing state."""
        if self._pending_sync:
            torch.mps.synchronize()
            self._pending_sync = False

    def __del__(self):
        """Ensure sync on cleanup to prevent crashes."""
        if hasattr(self, '_pending_sync') and self._pending_sync:
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Performs a single optimization step with async prefetching."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Sync from previous step's async page-out
        if self._pending_sync:
            torch.mps.synchronize()
            self._pending_sync = False

        has_mps_params = False

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            page_to_cpu = group['page_to_cpu']

            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if not params_with_grad:
                continue

            # Prefetch first param's states
            prefetched_exp_avg: Optional[torch.Tensor] = None
            prefetched_param: Optional[torch.Tensor] = None
            if page_to_cpu and params_with_grad:
                state = self.state[params_with_grad[0]]
                if len(state) > 0:
                    prefetched_exp_avg = state['exp_avg'].to(params_with_grad[0].device, non_blocking=True)
                    prefetched_param = params_with_grad[0]

            for i, p in enumerate(params_with_grad):
                grad = p.grad
                has_mps_params = has_mps_params or (p.device.type == 'mps')
                state = self.state[p]

                if len(state) == 0:
                    storage_device = 'cpu' if page_to_cpu else p.device
                    state['exp_avg'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=storage_device
                    )

                # Use prefetched state if available
                if prefetched_exp_avg is not None and prefetched_param is p:
                    exp_avg = prefetched_exp_avg
                elif page_to_cpu:
                    exp_avg = state['exp_avg'].to(p.device)
                else:
                    exp_avg = state['exp_avg']

                # Start prefetching NEXT param's states
                prefetched_exp_avg = None
                prefetched_param = None
                if page_to_cpu and i + 1 < len(params_with_grad):
                    next_p = params_with_grad[i + 1]
                    next_state = self.state[next_p]
                    if len(next_state) > 0:
                        prefetched_exp_avg = next_state['exp_avg'].to(next_p.device, non_blocking=True)
                        prefetched_param = next_p

                # Weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Lion update
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-group['lr'])

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                # Page out (async)
                if page_to_cpu:
                    state['exp_avg'] = exp_avg.to('cpu', non_blocking=True)

        if has_mps_params:
            self._pending_sync = True

        return loss
