from ..Module.Attention import softmax
import torch

def cross_entropy(logits, targets):
    # log(softmax(x)) = log(exp(x)/sum(exp(x))) = x - log_sum_exp
    # log_sum_exp = log(sum(exp(x))) = max(x) + log(sum(exp(x - max_x)))
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]  # (B, T, 1)
    shifted = logits - max_logits
    exp_shifted = torch.exp(shifted)
    log_sum_exp = torch.log(torch.sum(exp_shifted, dim=-1, keepdim=True))  # (B, T, 1)
    # log_softmax = logits - log_sum_exp
    log_probs = shifted - log_sum_exp

    # 取出真实标签位置的 log 概率
    targets_expanded = targets.unsqueeze(-1)  # (B, T, 1)
    log_prob_true = torch.gather(log_probs, dim=-1, index=targets_expanded)  # (B, T, 1)
    return -log_prob_true.mean()