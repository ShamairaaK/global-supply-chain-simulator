# global_supply_chain_v3.py
# Global Supply Chain Risk Simulator — Robust, CV-ready single-file app
# Save this file and run: streamlit run global_supply_chain_v3.py

import streamlit as st

# IMPORTANT: set_page_config must be the first Streamlit command in the file.
st.set_page_config(
    page_title="Global Supply Chain Risk Simulator (v3)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Now import the rest of the libraries
import pandas as pd
import numpy as np
import datetime as dt
import io
import zipfile
import plotly.express as px
import plotly.graph_objects as go

# yfinance is optional — if unavailable we fallback to synthetic data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

# Try openpyxl for Excel export; if unavailable we'll fallback to zip-of-csv
try:
    import openpyxl  # noqa: F401
    EXCEL_ENGINE = "openpyxl"
except Exception:
    EXCEL_ENGINE = None

# ------------------------
# Configuration / constants
# ------------------------
FOCUS_COMMODITIES = ["CrudeOil", "Wheat", "Copper", "NaturalGas", "RareEarths"]
# Best-effort Yahoo tickers (single-ticker downloads avoid multi-index issues)
YF_TICKERS = {
    "CrudeOil": "CL=F",
    "Wheat": "ZW=F",
    "Copper": "HG=F",
    "NaturalGas": "NG=F",
    "RareEarths": "REMX",
}
FX_TICKERS = {
    "USDINR": "INR=X",
    "USDEUR": "EUR=X",
    "USDCNY": "CNY=X",
}

# ------------------------
# Helper: safe single-ticker download
# ------------------------
def download_single_ticker_safe(ticker, period="3y"):
    """
    Download a single ticker via yfinance and return a DataFrame with columns ['date','price'].
    Returns None if data not available or yfinance not installed.
    """
    if not YFINANCE_AVAILABLE:
        return None
    try:
        df = yf.download(ticker, period=period, progress=False)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    # prefer 'Adj Close' else 'Close'
    if "Adj Close" in df.columns:
        series = df["Adj Close"].copy()
    elif "Close" in df.columns:
        series = df["Close"].copy()
    else:
        return None
    series.index = pd.to_datetime(series.index).date
    out = series.reset_index()
    out.columns = ["date", "price"]
    return out

# ------------------------
# Synthetic data generators (fallback / demo-ready)
# ------------------------
def synth_price_series(commodity, days=365 * 3, seed=None):
    """Create a synthetic daily price series for a commodity (used when live data unavailable)."""
    rng = np.random.default_rng(seed if seed is not None else abs(hash(commodity)) % (2 ** 32))
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=days, freq="D")
    base_levels = {
        "CrudeOil": 75.0,
        "Wheat": 6.0,
        "Copper": 3.8,
        "NaturalGas": 3.0,
        "RareEarths": 100.0,
    }
    vol_map = {
        "CrudeOil": 0.02,
        "Wheat": 0.018,
        "Copper": 0.015,
        "NaturalGas": 0.03,
        "RareEarths": 0.022,
    }
    price = base_levels.get(commodity, 10.0)
    sigma = vol_map.get(commodity, 0.02)
    rows = []
    for d in dates:
        shock = rng.normal(0, price * sigma)
        price = max(0.01, price + shock)
        rows.append({"date": d.date().isoformat(), "commodity": commodity, "price": round(price, 4)})
    return pd.DataFrame(rows)

def synth_fx_series(pair, days=365 * 3, seed=None):
    rng = np.random.default_rng(seed if seed is not None else abs(hash(pair)) % (2 ** 32))
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(end=end, periods=days, freq="D")
    base = {"USDINR": 75.0, "USDEUR": 0.92, "USDCNY": 6.7}.get(pair, 1.0)
    rows = []
    for d in dates:
        shock = rng.normal(0, max(0.0001, base * 0.0007))
        base = max(0.0001, base + shock)
        rows.append({"date": d.date().isoformat(), "pair": pair, "rate": round(base, 6)})
    return pd.DataFrame(rows)

def synth_trade_flows():
    """Create realistic-feeling trade flows table (no uploads required)."""
    countries = [
        "China",
        "India",
        "USA",
        "Germany",
        "Brazil",
        "Vietnam",
        "Chile",
        "Russia",
        "SaudiArabia",
        "Australia",
        "Peru",
    ]
    totals = {
        "CrudeOil": (2_500_000_000, ["SaudiArabia", "Russia", "USA"]),
        "Wheat": (800_000_000, ["USA", "Russia", "Australia"]),
        "Copper": (1_200_000_000, ["Chile", "Peru", "China"]),
        "NaturalGas": (1_000_000_000, ["USA", "Russia", "Australia"]),
        "RareEarths": (600_000_000, ["China", "Australia", "USA"]),
    }
    rng = np.random.default_rng(2025)
    rows = []
    for comm, (total, preferred) in totals.items():
        other = [c for c in countries if c not in preferred]
        n_extra = int(rng.integers(0, 3))
        chosen = list(preferred[:3]) + list(rng.choice(other, size=n_extra, replace=False))
        weights = rng.dirichlet(np.ones(len(chosen)) * 3.0)
        for c, w in zip(chosen, weights):
            rows.append(
                {
                    "commodity": comm,
                    "supplier_country": c,
                    "pct_dependency": float(w),
                    "annual_import_usd": float(round(total * w, 2)),
                }
            )
    return pd.DataFrame(rows)

# ------------------------
# Cached fetchers (safe)
# ------------------------
@st.cache_data(ttl=3600)
def get_price_tables():
    """
    Returns:
      df_comm: DataFrame with columns ['date','commodity','price']
      df_fx: DataFrame with columns ['date','pair','rate']
    Fetches live data per-ticker (safe single-ticker calls) and falls back to synthetic series if needed.
    """
    comm_parts = []
    fx_parts = []
    # commodities
    for comm, ticker in YF_TICKERS.items():
        df = download_single_ticker_safe(ticker, period="3y") if YFINANCE_AVAILABLE else None
        if df is not None:
            df["commodity"] = comm
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
            comm_parts.append(df.rename(columns={"price": "price"}))
    # if none were fetched, create synthetic for each
    if not comm_parts:
        for comm in FOCUS_COMMODITIES:
            comm_parts.append(synth_price_series(comm))
    df_comm = pd.concat(comm_parts, ignore_index=True)

    # fx
    for pair, ticker in FX_TICKERS.items():
        df = download_single_ticker_safe(ticker, period="3y") if YFINANCE_AVAILABLE else None
        if df is not None:
            # df has columns date, price (we will consider price -> rate)
            df = df.rename(columns={"price": "rate"})
            df["pair"] = pair
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
            fx_parts.append(df[["date", "pair", "rate"]])
    if not fx_parts:
        for pair in FX_TICKERS.keys():
            fx_parts.append(synth_fx_series(pair))
    df_fx = pd.concat(fx_parts, ignore_index=True)

    # normalize types
    df_comm["price"] = pd.to_numeric(df_comm["price"], errors="coerce")
    df_fx["rate"] = pd.to_numeric(df_fx["rate"], errors="coerce")
    return df_comm, df_fx

# ------------------------
# Analytics helpers
# ------------------------
def compute_hhi(df_trade):
    rows = []
    for comm, g in df_trade.groupby("commodity"):
        shares = g["pct_dependency"].astype(float).values
        hhi = float(np.sum(np.square(shares)))
        rows.append({"commodity": comm, "hhi": hhi})
    return pd.DataFrame(rows)

def rolling_volatility(df_comm, window=30):
    out = []
    df_comm = df_comm.copy()
    df_comm["date"] = pd.to_datetime(df_comm["date"])
    for comm, g in df_comm.groupby("commodity"):
        ts = g.sort_values("date").set_index("date")["price"].pct_change().dropna()
        if ts.shape[0] < window:
            continue
        roll = ts.rolling(window).std() * np.sqrt(252)
        dfv = roll.dropna().reset_index().rename(columns={0: "roll_vol"})
        dfv["commodity"] = comm
        dfv["roll_vol"] = roll.dropna().values
        out.append(dfv[["date", "commodity", "roll_vol"]])
    if out:
        return pd.concat(out, ignore_index=True)
    return pd.DataFrame(columns=["date", "commodity", "roll_vol"])

def correlation_matrix(df_comm):
    wide = df_comm.pivot_table(index="date", columns="commodity", values="price")
    rets = wide.pct_change().dropna(how="all")
    return rets.corr()

# ------------------------
# Monte Carlo price simulation (geometric returns)
# ------------------------
def monte_carlo_price_shock(series, mean_shift=0.0, vol_mult=1.0, n_sims=5000, horizon_days=252):
    """
    series: pd.Series of historical prices (ordered)
    mean_shift: fraction (e.g., 0.2 for +20% over horizon)
    vol_mult: multiplier to historical volatility
    Returns dictionary with final_prices array and metadata.
    """
    returns = series.pct_change().dropna()
    if returns.empty:
        mu = 0.0
        sigma = 0.01
    else:
        mu = returns.mean()
        sigma = returns.std()
    sigma *= vol_mult
    start_price = series.iloc[-1]
    rng = np.random.default_rng(123456)
    daily_shift = mean_shift / horizon_days
    sims = rng.normal(loc=(mu + daily_shift), scale=sigma, size=(n_sims, horizon_days))
    price_paths = start_price * np.exp(np.cumsum(sims, axis=1))
    final_prices = price_paths[:, -1]
    return {"final_prices": final_prices, "paths": price_paths, "start_price": start_price, "mu": mu, "sigma": sigma}

def generate_insight_text(commodity, supplier_country, dep_pct, sim_out):
    median = np.median(sim_out["final_prices"])
    p95 = np.percentile(sim_out["final_prices"], 95)
    p05 = np.percentile(sim_out["final_prices"], 5)
    tail_risk = (p95 - median) / max(1e-9, median)
    lines = [
        f"Commodity: {commodity}",
        f"Supplier: {supplier_country}",
        f"Dependency: {dep_pct*100:.1f}%",
        f"Baseline price: {sim_out.get('start_price', float('nan')):.2f}",
        f"Simulated median price (horizon): {median:.2f}",
        f"P05: {p05:.2f} | P95: {p95:.2f}",
        f"Tail risk (P95 over median): {tail_risk*100:.1f}%",
    ]
    return "\n".join(lines)

# ------------------------
# App UI
# ------------------------
def app():
    st.title("🌍 Global Supply Chain Risk Simulator — v3 (CV-ready)")
    st.markdown(
        """
        **What this demo shows:**
        - Commodity-focused risk analytics (no manual uploads required).  
        - Sourcing concentration (HHI), volatility, cross-commodity correlations.  
        - Monte Carlo stress tests showing price-tail risk and estimated import-cost impact.  
        - Downloadable simulation outputs and snapshot.
        """
    )

    # Load / prepare data
    df_comm, df_fx = get_price_tables()
    df_trade = synth_trade_flows()
    df_trade["pct_dependency"] = df_trade["pct_dependency"].astype(float)
    df_trade["annual_import_usd"] = df_trade["annual_import_usd"].astype(float)

    # Precompute analytics
    hhi = compute_hhi(df_trade)
    corr = correlation_matrix(df_comm)
    roll_vol = rolling_volatility(df_comm)

    # Sidebar controls
    st.sidebar.header("Simulation controls")
    commodity = st.sidebar.selectbox("Select commodity", FOCUS_COMMODITIES, index=0)
    suppliers = df_trade[df_trade["commodity"] == commodity].sort_values("pct_dependency", ascending=False)
    supplier = st.sidebar.selectbox("Supplier country", suppliers["supplier_country"].unique())
    dep_row = suppliers[suppliers["supplier_country"] == supplier].iloc[0]
    dep_pct = float(dep_row["pct_dependency"])
    baseline_import = float(dep_row["annual_import_usd"])

    st.sidebar.markdown(f"**Baseline annual imports from {supplier}:** ${baseline_import:,.0f}")
    st.sidebar.subheader("Stress parameters")
    price_shift_pct = st.sidebar.slider("Price mean shock (total %, horizon)", -50, 200, 20) / 100.0
    vol_multiplier = st.sidebar.slider("Volatility multiplier", 0.5, 3.0, 1.0, step=0.1)
    sims = st.sidebar.number_input("Monte Carlo sims", 500, 20000, 5000, step=500)

    # Tabs
    tab_overview, tab_comm, tab_trade, tab_stress, tab_insights = st.tabs(
        ["Overview", "Commodities", "Trade Risk", "Stress Tests", "Insights & Export"]
    )

    # --- Overview tab ---
    with tab_overview:
        st.header("Overview")
        col1, col2, col3 = st.columns(3)
        total_imports = df_trade.groupby("commodity")["annual_import_usd"].sum().reindex(FOCUS_COMMODITIES).fillna(0)
        col1.metric("Total annual imports (USD)", f"${total_imports.sum():,.0f}")
        top_comm = total_imports.idxmax()
        col2.metric("Largest imported commodity", top_comm)
        col3.metric("Avg supplier concentration (HHI)", f"{hhi['hhi'].mean():.2f}")

        st.markdown("**Import composition by commodity**")
        fig = px.bar(
            total_imports.reset_index(),
            x="commodity",
            y="annual_import_usd",
            labels={"annual_import_usd": "Annual import (USD)"},
            title="Annual import value by commodity",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Cross-commodity correlation (returns)** — useful to detect common shocks.")
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix (commodity returns)")
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- Commodities tab ---
    with tab_comm:
        st.header(f"{commodity} — time series & volatility (last 12 months)")
        comm_df = df_comm[df_comm["commodity"] == commodity].sort_values("date")
        if comm_df.empty:
            st.warning("Price data for this commodity is not available.")
        else:
            comm_df["date"] = pd.to_datetime(comm_df["date"])
            st.plotly_chart(px.line(comm_df.tail(365), x="date", y="price", title=f"{commodity} price (1Y)"), use_container_width=True)

            rv = roll_vol[roll_vol["commodity"] == commodity]
            if not rv.empty:
                rv["date"] = pd.to_datetime(rv["date"])
                st.plotly_chart(px.line(rv.tail(365), x="date", y="roll_vol", title="Rolling volatility (30D, annualized)"), use_container_width=True)
            else:
                st.info("Not enough historical data to compute rolling volatility for this commodity.")

            st.markdown("Supplier share (pie) & table")
            sup_df = suppliers.sort_values("annual_import_usd", ascending=False)
            st.plotly_chart(px.pie(sup_df, names="supplier_country", values="annual_import_usd", title=f"{commodity} supplier share"), use_container_width=True)
            st.dataframe(sup_df.reset_index(drop=True))

    # --- Trade Risk tab ---
    with tab_trade:
        st.header("Trade exposure & concentration")
        st.plotly_chart(px.bar(hhi.sort_values("hhi", ascending=False), x="commodity", y="hhi", title="HHI by commodity"), use_container_width=True)
        st.markdown("Top supplier exposures (dependency-weighted USD)")
        top_sup = df_trade.assign(risk=lambda d: d["pct_dependency"] * d["annual_import_usd"]).sort_values("risk", ascending=False).head(10)
        st.plotly_chart(px.bar(top_sup, x="supplier_country", y="risk", color="commodity", title="Top supplier exposures"), use_container_width=True)
        st.dataframe(top_sup.reset_index(drop=True))

    # --- Stress Tests tab ---
    with tab_stress:
        st.header("Monte Carlo stress testing")
        st.markdown(f"Simulating a **{price_shift_pct*100:.1f}%** mean price shift for **{commodity}** with vol multiplier **{vol_multiplier:.2f}**.")
        series = df_comm[df_comm["commodity"] == commodity].sort_values("date")["price"].reset_index(drop=True)
        if series.empty:
            st.warning("No price series available for simulations.")
        else:
            run = st.button("Run Monte Carlo")
            if run:
                with st.spinner("Running simulations..."):
                    sim = monte_carlo_price_shock(series, mean_shift=price_shift_pct, vol_mult=vol_multiplier, n_sims=int(sims), horizon_days=252)
                median_final = np.median(sim["final_prices"])
                p95 = np.percentile(sim["final_prices"], 95)
                p05 = np.percentile(sim["final_prices"], 5)

                c1, c2, c3 = st.columns(3)
                c1.metric("Start price", f"{sim['start_price']:.2f}")
                c2.metric("Median final price (horizon)", f"{median_final:.2f}")
                c3.metric("P95 final price", f"{p95:.2f}")

                # distribution plot
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=sim["final_prices"], nbinsx=80, name="Final prices"))
                fig_hist.update_layout(title=f"Distribution of simulated final {commodity} prices", xaxis_title="Price", yaxis_title="Count")
                st.plotly_chart(fig_hist, use_container_width=True)

                # translate to import-cost impact (simple proportional model)
                start = sim["start_price"]
                mean_price = np.mean(sim["final_prices"])
                new_expected_import = baseline_import * (mean_price / start)
                delta = new_expected_import - baseline_import
                st.subheader("Estimated import-cost impact (simple proportional model)")
                st.markdown(f"- Baseline annual imports from **{supplier}**: **${baseline_import:,.0f}**")
                st.markdown(f"- Expected annual imports under simulated mean price: **${new_expected_import:,.0f}** (Δ = **${delta:,.0f}**)")

                # Download simulated final prices CSV
                sample_df = pd.DataFrame({"final_price": sim["final_prices"]})
                st.download_button("Download simulated final prices (CSV)", sample_df.to_csv(index=False).encode("utf-8"), file_name=f"{commodity}_sim_final_prices.csv", mime="text/csv")

    # --- Insights & Export tab ---
    with tab_insights:
        st.header("Analyst-style insights (neutral)")
        top_hhi = hhi.sort_values("hhi", ascending=False).head(3)
        st.markdown("**Top concentrated commodities (HHI)**")
        for _, r in top_hhi.iterrows():
            st.markdown(f"- {r['commodity']}: HHI = {r['hhi']:.3f}")

        if not roll_vol.empty:
            vol_mean = roll_vol.groupby("commodity")["roll_vol"].mean().sort_values(ascending=False)
            top_vol = vol_mean.index[0]
            st.markdown(f"**Highest avg rolling volatility:** {top_vol} (avg = {vol_mean.iloc[0]:.3f})")
        else:
            st.markdown("Volatility metrics unavailable (insufficient data).")

        st.markdown("**Top supplier risk (dependency-weighted USD)**")
        top_risk = df_trade.assign(risk=lambda d: d["pct_dependency"] * d["annual_import_usd"]).sort_values("risk", ascending=False).head(6)
        st.dataframe(top_risk.reset_index(drop=True))

        st.markdown("---")
        st.subheader("Export consolidated snapshot")
        if EXCEL_ENGINE is not None:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine=EXCEL_ENGINE) as writer:
                df_trade.to_excel(writer, sheet_name="trade_flows", index=False)
                df_comm.tail(2000).to_excel(writer, sheet_name="commodity_prices", index=False)
                df_fx.tail(2000).to_excel(writer, sheet_name="fx_rates", index=False)
                hhi.to_excel(writer, sheet_name="hhi", index=False)
            st.download_button("Download snapshot (XLSX)", data=buf.getvalue(), file_name="supply_chain_snapshot.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            # fallback: package CSVs into a ZIP for convenience (no extra install required)
            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, mode="w") as zf:
                zf.writestr("trade_flows.csv", df_trade.to_csv(index=False))
                zf.writestr("commodity_prices.csv", df_comm.to_csv(index=False))
                zf.writestr("fx_rates.csv", df_fx.to_csv(index=False))
                zf.writestr("hhi.csv", hhi.to_csv(index=False))
            st.info("openpyxl not installed — providing CSV snapshot as a ZIP file.")
            st.download_button("Download snapshot (ZIP of CSVs)", data=zbuf.getvalue(), file_name="supply_chain_snapshot.zip", mime="application/zip")

    # Footer / developer notes (helpful for recruiters)
    st.markdown("---")
    st.caption(
        "Developer notes: This demo auto-generates trade flows and fetches live prices when available. "
        "For production: replace synthetic flows with UN Comtrade extracts, add lead-time/inventory models, "
        "and link an optimization routine to recommend sourcing shifts or hedging notional amounts."
    )


if __name__ == "__main__":
    app()
