import torch

def gradient_clipping(parameters, M):
    eps = 1e-6

    grads = [p.grad.data for p in parameters if p.grad is not None]
    
    # If no gradients, return early
    if len(grads) == 0:
        return
    
    # Compute the global L2 norm of all gradients
    grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]), p=2)
    
    if grad_norm <= M:
        return
    
    # M / (||g||_2 + eps)
    scale_factor = M / (grad_norm + eps)
    
    # Apply scaling in place
    for p in parameters:
        if p.grad is not None:
            p.grad.data.mul_(scale_factor)
    return