"""Monte Carlo option pricing (GBM and GBM + Jumps)."""
import numpy as np


def monte_carlo_price(S, K, T, r, sigma, option_type="call",
                      n_sims=50000, jump=False, lam=0.5, mu_j=-0.1, sigma_j=0.2):
    """
    Monte Carlo European option pricer.
    If jump=True, simulates GBM + Poisson jump-diffusion.
    Returns (price, std_error, terminal_prices).
    """
    if T <= 0 or S <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        return intrinsic, 0.0, np.array([S])

    Z = np.random.standard_normal(n_sims)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z

    if jump:
        n_jumps = np.random.poisson(lam * T, n_sims)
        jump_sizes = np.array([
            np.sum(np.random.normal(mu_j, sigma_j, nj)) if nj > 0 else 0.0
            for nj in n_jumps
        ])
        drift_adj = drift - lam * mu_j * T
        ST = S * np.exp(drift_adj + diffusion + jump_sizes)
    else:
        ST = S * np.exp(drift + diffusion)

    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
    return price, std_err, ST
