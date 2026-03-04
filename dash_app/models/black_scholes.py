"""Black-Scholes pricing and Greeks computation."""
import numpy as np
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes European option price."""
    if sigma <= 0 or T <= 0 or S <= 0:
        if option_type == "call":
            return max(S - K * np.exp(-r * T), 0)
        return max(K * np.exp(-r * T) - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_greeks(S, K, T, r, sigma, option_type="call"):
    """Compute BS Greeks: delta, gamma, theta, vega, rho."""
    if sigma <= 0 or T <= 0 or S <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -S * pdf_d1 * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -S * pdf_d1 * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T) / 100
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}


def bachelier(S, K, T, r, sigma_n, option_type="call"):
    """Bachelier (Normal) model — useful when S close to 0."""
    sigma_abs = sigma_n * S
    if sigma_abs <= 0 or T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d = (S - K) / (sigma_abs * np.sqrt(T))
    if option_type == "call":
        return np.exp(-r * T) * (
            (S - K) * norm.cdf(d) + sigma_abs * np.sqrt(T) * norm.pdf(d)
        )
    return np.exp(-r * T) * (
        (K - S) * norm.cdf(-d) + sigma_abs * np.sqrt(T) * norm.pdf(d)
    )


def binomial_tree(S, K, T, r, sigma, option_type="call", N=500):
    """CRR Binomial Tree pricing (supports American exercise)."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    ST = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N + 1, 1))
    if option_type == "call":
        V = np.maximum(ST - K, 0)
    else:
        V = np.maximum(K - ST, 0)
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
    return V[0]
