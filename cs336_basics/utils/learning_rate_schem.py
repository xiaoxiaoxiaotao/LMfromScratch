import math

def learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):

    if t < T_w:
        # Warm-up phase
        return (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        # Cosine annealing phase
        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (1 + math.cos(progress * math.pi)) * (alpha_max - alpha_min)
    else:
        # Post-annealing phase
        return alpha_min