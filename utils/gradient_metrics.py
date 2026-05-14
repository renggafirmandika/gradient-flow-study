# utils/gradient_metrics.py
import torch
from collections import defaultdict

class GradientTracker:
    """
    Attaches hooks to a model to record gradient norms
    at each named parameter after every backward pass.
    
    Usage:
        tracker = GradientTracker(model)
        loss.backward()
        metrics = tracker.get_metrics()
        tracker.reset()
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._grad_norms = defaultdict(list)
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Capture name in closure
                def make_hook(param_name):
                    def hook(grad):
                        if grad is not None:
                            self._grad_norms[param_name].append(grad.norm().item())
                    return hook
                handle = param.register_hook(make_hook(name))
                self._hooks.append(handle)

    def reset(self):
        self._grad_norms = defaultdict(list)

    def get_metrics(self) -> dict:
        """
        Returns per-parameter gradient norms, plus summary stats:
        - mean_grad_norm: average across all parameters
        - min_grad_norm: smallest (most vanished) parameter gradient
        - max_grad_norm: largest (most exploded) parameter gradient
        - vanishing_ratio: fraction of parameters with norm < 1e-4
        """
        if not self._grad_norms:
            return {}

        all_norms = []
        per_param = {}
        for name, norms in self._grad_norms.items():
            avg = sum(norms) / len(norms)
            per_param[name] = avg
            all_norms.append(avg)

        return {
            'per_param': per_param,
            'mean_grad_norm': sum(all_norms) / len(all_norms),
            'min_grad_norm': min(all_norms),
            'max_grad_norm': max(all_norms),
            'vanishing_ratio': sum(1 for n in all_norms if n < 1e-4) / len(all_norms)
        }

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()