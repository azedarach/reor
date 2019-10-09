"""
Provides routines for perfoming SPG descent optimization.
"""

def get_next_spg_step_length(current_step_length, delta, f_old, f_new,
                             sigma_one=0.1, sigma_two=0.9):
    """Return next step length for line search."""

    step_length_tmp = (-0.5 * current_step_length ** 2 * delta /
                       (f_new - f_old - current_step_length * delta))

    next_step_length = 0
    if sigma_one <= step_length_tmp <= sigma_two * current_step_length:
        next_step_length = step_length_tmp
    else:
        next_step_length = 0.5 * current_step_length

    return next_step_length


def get_next_spg_alpha(beta, sksk, alpha_min=1e-3, alpha_max=1e3):
    """Return next value of alpha in SPG optimization."""

    if beta <= 0:
        return alpha_max

    return min(alpha_max, max(alpha_min, sksk / beta))
