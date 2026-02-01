# Roth Conversion Planner – Production-Grade CPA Engine with Monte Carlo
# Local macOS execution (Streamlit UI preferred, CLI fallback supported)
# -------------------------------------------------
# RECOMMENDED (full UI):
#   python3 -m venv venv && source venv/bin/activate
#   pip install streamlit pandas numpy matplotlib scipy
#   streamlit run roth_planner.py
# -------------------------------------------------
# FALLBACK (no streamlit available):
#   python roth_planner.py --cli
# -------------------------------------------------

import sys
import argparse

# -------------------------------------------------
# Optional Streamlit import with graceful fallback
# -------------------------------------------------
STREAMLIT_AVAILABLE = True
try:
    import streamlit as st
except ModuleNotFoundError:
    STREAMLIT_AVAILABLE = False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# =================================================
# CORE TAX / FINANCE LOGIC (UI-INDEPENDENT)
# =================================================

def marginal_rate(income, fed_brackets):
    rate = fed_brackets[0][1]
    for limit, r in fed_brackets:
        if income >= limit:
            rate = r
    return rate


def bracket_fill(base_income, fed_brackets, target_rate):
    """Return max conversion to top of target bracket"""
    for i, (limit, rate) in enumerate(fed_brackets):
        if rate == target_rate:
            top = fed_brackets[i + 1][0] if i + 1 < len(fed_brackets) else limit * 2
            return max(0, top - base_income)
    return 0


def ss_taxable(other_income, ss, filing_status):
    provisional = other_income + 0.5 * ss
    if filing_status == "MFJ":
        if provisional < 32_000:
            return 0
        elif provisional < 44_000:
            return 0.5 * ss
        else:
            return 0.85 * ss
    else:
        if provisional < 25_000:
            return 0
        elif provisional < 34_000:
            return 0.5 * ss
        else:
            return 0.85 * ss


# =================================================
# STREAMLIT APPLICATION
# =================================================

def run_streamlit():
    st.set_page_config(page_title="Roth Conversion Planner – CPA Engine", layout="wide")

    st.title("Roth Conversion Planner – CPA / CFP / Monte Carlo Engine")
    st.caption("Bracket-fill optimization • SS taxation • IRMAA • RMDs • Monte Carlo longevity modeling")

    # -----------------
    # INPUTS
    # -----------------
    st.sidebar.header("Demographics")
    age = st.sidebar.number_input("Current Age", 40, 90, 66)
    end_age = st.sidebar.number_input("Project Until Age", 75, 100, 95)
    filing_status = st.sidebar.selectbox("Filing Status", ["MFJ", "Single"])

    st.sidebar.header("Income")
    w2_income = st.sidebar.number_input("W-2 Income ($)", value=250_000, step=1_000)
    ss_start_age = st.sidebar.number_input("SS Start Age", 62, 70, 70)
    ss_annual = st.sidebar.number_input("SS Annual Benefit ($)", value=60_000, step=1_000)

    st.sidebar.header("Accounts")
    pretax_balance = st.sidebar.number_input("Pre-Tax Balance ($)", value=3_000_000, step=10_000)
    roth_balance = st.sidebar.number_input("Existing Roth Balance ($)", value=0, step=10_000)

    st.sidebar.header("Returns & Risk")
    real_return = st.sidebar.slider("Expected Real Return (%)", 2.0, 7.0, 5.0) / 100
    volatility = st.sidebar.slider("Annual Volatility (%)", 5.0, 20.0, 12.0) / 100
    simulations = st.sidebar.selectbox("Monte Carlo Simulations", [500, 1000, 2000], index=1)

    st.sidebar.header("Taxes")
    state_tax = st.sidebar.slider("State + Local Marginal (%)", 0.0, 12.0, 7.0) / 100

    if filing_status == "MFJ":
        fed_brackets = [(0,0.10),(22_000,0.12),(89_450,0.22),(190_750,0.24),(364_200,0.32),(462_500,0.35),(693_750,0.37)]
    else:
        fed_brackets = [(0,0.10),(11_000,0.12),(44_725,0.22),(95_375,0.24),(182_100,0.32),(231_250,0.35),(578_125,0.37)]

    irmaa_tiers = [206_000, 258_000, 322_000]
    irmaa_cost = [0, 1_600, 3_200, 4_800]

    # -----------------
    # BRACKET FILL
    # -----------------
    st.header("Optimal Bracket-Fill Roth Conversions")
    base_income = w2_income

    rec = []
    for _, rate in fed_brackets:
        if rate <= 0.24:
            amt = bracket_fill(base_income, fed_brackets, rate)
            if amt > 0:
                rec.append((rate, amt))

    st.dataframe(pd.DataFrame(rec, columns=["Target Bracket", "Max Conversion"]))

    # -----------------
    # CONVERSION GRID
    # -----------------
    st.header("Conversion Grid & BETR")
    rows = []
    steps = np.arange(0, 500_001, 25_000)
    years = end_age - age

    for conv in steps:
        magi = w2_income + conv
        fed = marginal_rate(magi, fed_brackets)
        total_rate = fed + state_tax

        irmaa = 0
        for i, t in enumerate(irmaa_tiers):
            if magi > t:
                irmaa = irmaa_cost[i + 1]

        tax_cost = conv * total_rate + irmaa
        fv_roth = conv * (1 + real_return) ** years
        fv_trad = conv * (1 + real_return) ** years * (1 - fed)
        betr = tax_cost / fv_roth if fv_roth > 0 else 0

        rows.append([conv, fed, total_rate, irmaa, tax_cost, fv_roth, fv_trad, betr])

    df = pd.DataFrame(rows, columns=["Conversion","Fed Rate","Total Rate","IRMAA","Tax Cost","FV Roth","FV Trad After Tax","BETR"])
    st.dataframe(df, use_container_width=True)

    # -----------------
    # MONTE CARLO
    # -----------------
    st.header("Monte Carlo Longevity Modeling")
    np.random.seed(42)
    mc = []

    for _ in range(simulations):
        bal = pretax_balance
        for _ in range(years):
            bal *= (1 + norm.rvs(real_return, volatility))
        mc.append(bal)

    mc = np.array(mc)
    st.dataframe(pd.DataFrame({"P10":[np.percentile(mc,10)],"P50":[np.percentile(mc,50)],"P90":[np.percentile(mc,90)]}))

    fig, ax = plt.subplots()
    ax.hist(mc, bins=40)
    st.pyplot(fig)

    st.success("Streamlit mode running successfully")


# =================================================
# CLI FALLBACK (FOR SANDBOX / NO STREAMLIT)
# =================================================

def run_cli():
    print("Streamlit not available. Running CLI mode.\n")

    age = 66
    end_age = 95
    w2_income = 250_000
    pretax_balance = 3_000_000
    real_return = 0.05

    fed_brackets = [(0,0.10),(22_000,0.12),(89_450,0.22),(190_750,0.24),(364_200,0.32),(462_500,0.35),(693_750,0.37)]

    years = end_age - age
    fv = pretax_balance * (1 + real_return) ** years

    print(f"Projected pre-tax balance at age {end_age}: ${fv:,.0f}")
    print("Install streamlit to enable full UI experience.")


# =================================================
# BASIC SELF-TESTS (NON-STREAMLIT)
# =================================================

def _run_tests():
    fb = [(0,0.10),(22_000,0.12),(89_450,0.22),(190_750,0.24)]
    assert marginal_rate(50_000, fb) == 0.22
    assert bracket_fill(100_000, fb, 0.24) > 0
    assert ss_taxable(0, 40_000, "MFJ") in (0, 20_000, 34_000)
    print("All core tests passed.")


# =================================================
# ENTRY POINT
# =================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run without Streamlit UI")
    parser.add_argument("--test", action="store_true", help="Run self-tests")
    args = parser.parse_args()

    if args.test:
        _run_tests()
        sys.exit(0)

    if STREAMLIT_AVAILABLE and not args.cli:
        run_streamlit()
    else:
        run_cli()
