"""Merton Jump-Diffusion model."""
import numpy as np
from scipy.stats import norm
from math import factorial


def merton_jump_diffusion(S, K, T, r, sigma, lam, mu_j, sigma_j, option_type="call", n_terms=50):
    """
    Merton (1976) jump-diffusion: GBM + Poisson jumps.
    λ = jump intensity, μⱼ = mean jump size, σⱼ = jump vol.
    """
    if sigma <= 0 or T <= 0 or S <= 0:
        if option_type == "call":
            return max(S - K * np.exp(-r * T), 0)
        return max(K * np.exp(-r * T) - S, 0)

    price = 0.0
    lam_prime = lam * (1 + mu_j)
    for n in range(n_terms):
        sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T) if T > 0 else sigma
        r_n = r - lam * mu_j + n * np.log(1 + mu_j) / T
        poisson_weight = np.exp(-lam_prime * T) * (lam_prime * T) ** n / factorial(n)

        d1 = (np.log(S / K) + (r_n + 0.5 * sigma_n**2) * T) / (sigma_n * np.sqrt(T))
        d2 = d1 - sigma_n * np.sqrt(T)

        if option_type == "call":
            bs_n = S * norm.cdf(d1) - K * np.exp(-r_n * T) * norm.cdf(d2)
        else:
            bs_n = K * np.exp(-r_n * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        price += poisson_weight * bs_n

    return price
