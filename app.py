
# app.py — Monte Carlo Options Pricer (Pro, Branded)
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
from io import BytesIO

st.set_page_config(page_title="RS&Co | Monte Carlo Options Pricer (Pro)", page_icon="📊", layout="wide")

BRAND_NAME = "R Subramanian and Company LLP"
TAGLINE = "Professional Monte Carlo Options Pricer — European, Asian, Barrier + Greeks"
FOOTER_NOTE = "Prepared by R Subramanian and Company LLP — Confidential"

c1, c2 = st.columns([1, 6])
with c1:
    st.image("assets/logo_placeholder.png", width=92)
with c2:
    st.markdown(f"## **{BRAND_NAME}**")
    st.markdown(f"#### {TAGLINE}")
st.divider()

def bs_price_call_put(S0, K, r, sigma, T, option_type="Call"):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S0 - K, 0.0) if option_type == "Call" else max(K - S0, 0.0)
        return intrinsic
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def simulate_paths(S0, r, sigma, T, N, M, antithetic=False, seed=None):
    rng = np.random.default_rng(seed)
    dt = T / M
    mu = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    if antithetic:
        n_half = N // 2
        Z_half = rng.standard_normal((n_half, M))
        Z = np.vstack([Z_half, -Z_half])
        if Z.shape[0] < N:
            Z = np.vstack([Z, rng.standard_normal((1, M))])
    else:
        Z = rng.standard_normal((N, M))
    increments = mu + vol * Z
    log_rel = np.cumsum(increments, axis=1)
    rel = np.exp(log_rel)
    paths = np.concatenate([np.ones((rel.shape[0], 1)), rel], axis=1) * S0
    return paths[:N, :]

def payoff_from_paths(paths, K, option_type, payoff_kind, barrier_type=None, barrier_level=None):
    ST = paths[:, -1]
    if payoff_kind == "European":
        payoff = np.maximum(ST - K, 0.0) if option_type == "Call" else np.maximum(K - ST, 0.0)
        return payoff, ST
    if payoff_kind == "Asian (Arithmetic Avg)":
        avg = paths[:, 1:].mean(axis=1)
        payoff = np.maximum(avg - K, 0.0) if option_type == "Call" else np.maximum(K - avg, 0.0)
        return payoff, avg
    if payoff_kind == "Barrier":
        path_min = paths.min(axis=1)
        path_max = paths.max(axis=1)
        crossed_up = path_max >= barrier_level
        crossed_down = path_min <= barrier_level
        base = np.maximum(ST - K, 0.0) if option_type == "Call" else np.maximum(K - ST, 0.0)
        if barrier_type == "Up-and-Out":
            payoff = np.where(~crossed_up, base, 0.0)
        elif barrier_type == "Down-and-Out":
            payoff = np.where(~crossed_down, base, 0.0)
        elif barrier_type == "Up-and-In":
            payoff = np.where(crossed_up, base, 0.0)
        elif barrier_type == "Down-and-In":
            payoff = np.where(crossed_down, base, 0.0)
        else:
            raise ValueError("Unknown barrier type")
        return payoff, ST
    raise ValueError("Unknown payoff kind")

def mc_engine(S0, K, r, sigma, T, N, M, option_type, payoff_kind,
              antithetic=True, control_variate=False,
              barrier_type=None, barrier_level=None, seed=None):
    paths = simulate_paths(S0, r, sigma, T, N, M, antithetic=antithetic, seed=seed)
    payoff, x_metric = payoff_from_paths(paths, K, option_type, payoff_kind, barrier_type, barrier_level)
    disc = np.exp(-r * T)
    price_naive = disc * payoff.mean()
    std_payoff = payoff.std(ddof=1)
    se_naive = disc * std_payoff / np.sqrt(N)
    itm_prob = float((payoff > 0).mean())
    price = price_naive; se = se_naive
    if control_variate:
        if payoff_kind == "Asian (Arithmetic Avg)":
            X = disc * x_metric; EX = S0
        else:
            X = disc * paths[:, -1]; EX = S0
        Y = disc * payoff
        var_X = X.var(ddof=1)
        if var_X > 0:
            beta = np.cov(Y, X, ddof=1)[0,1] / var_X
            price = float(Y.mean() - beta * (X.mean() - EX))
            se = float((Y - beta*X).std(ddof=1) / np.sqrt(N))
    ci_low = price - 1.96*se; ci_high = price + 1.96*se
    return {"price": float(price), "se": float(se), "ci_low": float(ci_low), "ci_high": float(ci_high),
            "itm_prob": itm_prob, "x_metric": x_metric, "payoff": payoff}

# Sidebar
with st.sidebar:
    st.header("Inputs")
    S0 = st.number_input("Spot (S₀)", min_value=0.01, value=100.0, step=1.0, format="%.4f")
    K = st.number_input("Strike (K)", min_value=0.01, value=110.0, step=1.0, format="%.4f")
    T = st.number_input("Time to Expiry (years, T)", min_value=0.0001, value=1.0, step=0.25, format="%.4f")
    r_pct = st.number_input("Risk-free rate r (%)", min_value=0.0, value=6.0, step=0.25, format="%.4f")
    sigma_pct = st.number_input("Volatility σ (%)", min_value=0.01, value=25.0, step=0.25, format="%.4f")
    option_type = st.radio("Option type", ["Call", "Put"], horizontal=True)
    payoff_kind = st.selectbox("Payoff type", ["European", "Asian (Arithmetic Avg)", "Barrier"])
    M = st.selectbox("Steps per path (M)", options=[1, 12, 52, 252], index=0)
    N = st.number_input("Simulations (N)", min_value=5000, value=50000, step=5000)
    if payoff_kind == "Barrier":
        barrier_type = st.selectbox("Barrier type", ["Up-and-Out", "Down-and-Out", "Up-and-In", "Down-and-In"])
        barrier_level = st.number_input("Barrier level (B)", min_value=0.01, value=120.0, step=1.0, format="%.4f")
    else:
        barrier_type = None; barrier_level = None

    st.markdown("---")
    st.subheader("Advanced")
    antithetic = st.checkbox("Use Antithetic Variates (faster convergence)", value=True)
    control_variate = st.checkbox("Use Control Variate", value=False)
    use_seed = st.checkbox("Lock random seed (reproducible)", value=True)
    seed_val = st.number_input("Random Seed", min_value=0, value=42, step=1)

    st.markdown("---")
    with st.expander("Client explanation (plain-English)", expanded=False):
        st.markdown("""
        • **European:** Only the **final** price at expiry matters.  
        • **Asian:** The **average** price over time matters (smoother).  
        • **Barrier:** Option can **switch on/off** if a level is crossed.  
        **Price =** discounted **average** payoff across many futures.  
        **SE / 95% CI** show precision.
        """)

r = r_pct/100.0; sigma = sigma_pct/100.0; seed = int(seed_val) if use_seed else None

st.info("Price = discounted average payoff across simulated futures. Smaller SE / tighter CI = more precise estimate.")
if st.button("Run Pricing Simulation", type="primary", use_container_width=True):
    out = mc_engine(S0, K, r, sigma, T, int(N), int(M), option_type, payoff_kind,
                    antithetic, control_variate, barrier_type, barrier_level, seed)
    price, se, ci_low, ci_high = out["price"], out["se"], out["ci_low"], out["ci_high"]
    itm_prob = out["itm_prob"]; x_metric = out["x_metric"]; payoff = out["payoff"]
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Option Price", f"{price:,.4f}")
    k2.metric("Std. Error (SE)", f"{se:,.6f}")
    k3.metric("95% CI Low", f"{ci_low:,.4f}")
    k4.metric("95% CI High", f"{ci_high:,.4f}")
    k5.metric("ITM Probability", f"{100*itm_prob:,.2f}%")
    st.divider()
    label = "Terminal Price S_T" if payoff_kind != "Asian (Arithmetic Avg)" else "Average Price (Asian)"
    st.subheader(f"Distribution of {label}")
    hist_vals, bin_edges = np.histogram(x_metric, bins=50)
    hist_df = pd.DataFrame({"bin_left": bin_edges[:-1], "count": hist_vals})
    st.bar_chart(hist_df.set_index("bin_left"))
    st.subheader("Payoff vs Price Metric")
    sample_n = min(5000, len(x_metric))
    idx = np.random.choice(len(x_metric), size=sample_n, replace=False)
    scatter_df = pd.DataFrame({label: x_metric[idx], "Payoff": payoff[idx]})
    st.scatter_chart(scatter_df, x=label, y="Payoff")
