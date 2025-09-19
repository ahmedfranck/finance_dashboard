# Advanced Fintech Interactive Dashboard (fictional data)
# Audience: Sales, Operations, Risk & Fraud, Finance/Leadership, Customer, Board
# Streamlit deploy ready

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Fintech Performance Cockpit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Utilities
# -----------------------------
@st.cache_data(show_spinner=False)
def make_data(seed=42, months=18, n_customers=12000):
    """
    Generate fictional fintech data:
    - Customers, subscriptions, monthly recurring revenue, transactions, refunds, fraud, chargebacks
    - Regions, products, plans, sales reps
    """
    rng = np.random.default_rng(seed)

    # Dates
    today = pd.to_datetime(datetime.utcnow().date())
    start = (today.replace(day=1) - relativedelta(months=months-1))
    days = pd.date_range(start, today, freq="D")

    # Dimensions
    regions = ["Senegal", "Côte d'Ivoire", "Ghana", "Nigeria", "Mali"]
    products = ["Wallet", "Paylink", "Payouts", "POS", "BNPL"]
    plans = ["Starter", "Growth", "Scale", "Enterprise"]
    segments = ["SME", "Mid Market", "Enterprise", "Consumer"]
    industries = ["Retail", "Services", "Food", "Logistics", "Healthcare", "Education"]
    reps = [f"Rep_{i:02d}" for i in range(1, 21)]

    # Customers
    cust = pd.DataFrame({
        "customer_id": np.arange(1, n_customers+1),
        "region": rng.choice(regions, n_customomers:=n_customers, p=[0.25, 0.20, 0.18, 0.25, 0.12]),
        "segment": rng.choice(segments, n_customomers, p=[0.45, 0.25, 0.10, 0.20]),
        "industry": rng.choice(industries, n_customomers),
        "acquired_date": rng.choice(pd.date_range(start, today - pd.Timedelta(days=30), freq="D"), n_customomers),
        "sales_rep": rng.choice(reps, n_customomers),
        "product_primary": rng.choice(products, n_customomers, p=[0.35, 0.15, 0.18, 0.20, 0.12]),
    })
    # Plan assignment influenced by segment
    seg_to_plan_probs = {
        "Consumer": [0.75, 0.20, 0.04, 0.01],
        "SME": [0.45, 0.35, 0.15, 0.05],
        "Mid Market": [0.15, 0.45, 0.30, 0.10],
        "Enterprise": [0.02, 0.18, 0.40, 0.40],
    }
    cust["plan"] = cust["segment"].map(lambda s: rng.choice(plans, p=seg_to_plan_probs[s]))

    # Pricing baseline per plan (MRR)
    plan_mrr = {"Starter": 25, "Growth": 79, "Scale": 249, "Enterprise": 950}

    # Intensity multipliers
    prod_intensity = {"Wallet": 1.2, "Paylink": 1.0, "Payouts": 0.8, "POS": 1.4, "BNPL": 0.6}
    plan_mult = {"Starter": 0.6, "Growth": 1.0, "Scale": 1.7, "Enterprise": 3.0}

    # Churn curves
    base_churn_monthly = {"Starter": 0.035, "Growth": 0.025, "Scale": 0.018, "Enterprise": 0.011}

    # Cohorts by month
    cust["cohort_month"] = cust["acquired_date"].values.astype("datetime64[M]")
    months_index = pd.period_range(start=start.to_period('M'), end=today.to_period('M'), freq="M").to_timestamp()

    # Subscription status over time
    subs = []
    for _, row in cust.iterrows():
        start_date = row["acquired_date"].replace(day=1)
        active = True
        for m in months_index:
            if m < start_date:
                continue
            if active and m > start_date and rng.random() < base_churn_monthly[row["plan"]]:
                active = False
            subs.append({
                "customer_id": row["customer_id"],
                "month": m,
                "active": int(active),
                "plan": row["plan"],
                "region": row["region"],
                "segment": row["segment"],
                "product_primary": row["product_primary"],
                "sales_rep": row["sales_rep"],
                # small uplift for POS-heavy accounts
                "mrr": plan_mrr[row["plan"]] * (1 + 0.15 * (row["product_primary"] in ["POS"]))
            })
    subs = pd.DataFrame(subs)
    subs.loc[subs["active"] == 1, "mrr"] *= (1 + 0.004 * (subs["month"] - subs["month"].min()).dt.days / 30)
    subs["mrr"] = subs["mrr"].round(2)

    # Transactions table (aggregate daily)
    txns = []
    for m in months_index:
        month_days = pd.date_range(m, m + relativedelta(months=1) - timedelta(days=1), freq="D")
        # >>> FIX: include sales_rep in active_ids so it flows into txns
        active_ids = subs.loc[
            (subs["month"] == m) & (subs["active"] == 1),
            ["customer_id", "product_primary", "plan", "region", "segment", "sales_rep"]
        ]
        if active_ids.empty:
            continue

        # baseline per customer per day
        base = 0.2 + 1.2 * active_ids["segment"].map({"Consumer":1.0, "SME":1.2, "Mid Market":1.5, "Enterprise":1.8}).values
        base *= active_ids["product_primary"].map(prod_intensity).values
        base *= active_ids["plan"].map(plan_mult).values
        base = np.clip(base, 0.2, None)

        for d in month_days:
            dfac = 0.9 + rng.random(len(active_ids)) * 0.4
            counts = rng.poisson(lam=base * dfac)

            mean_amt = 22 + 18 * rng.random(len(active_ids))
            amt = np.maximum(rng.normal(mean_amt, 6), 3)
            gross = counts * amt

            fraud_rate = 0.0025 + 0.002 * (d.weekday() in [4,5])
            refunds_rate = 0.01 + 0.01 * rng.random(len(active_ids))
            fraud = (rng.random(len(active_ids)) < fraud_rate) * (amt * (1 + rng.random(len(active_ids))*4))
            refunds = (counts * refunds_rate * amt) * (rng.random(len(active_ids)) < 0.6)
            net = gross - refunds - fraud

            chunk = active_ids.copy()
            chunk["date"] = d
            chunk["txn_count"] = counts
            chunk["gross_revenue"] = gross
            chunk["refunds"] = refunds
            chunk["fraud_loss"] = fraud
            chunk["net_revenue"] = net
            txns.append(chunk)

    txns = pd.concat(txns, ignore_index=True) if txns else pd.DataFrame(
        columns=["customer_id","product_primary","plan","region","segment","sales_rep","date",
                 "txn_count","gross_revenue","refunds","fraud_loss","net_revenue"]
    )
    for c in ["gross_revenue","refunds","fraud_loss","net_revenue"]:
        txns[c] = txns[c].astype(float).round(2) if not txns.empty else txns[c]

    # Funnel (fictional)
    funnel = []
    for m in months_index:
        leads = int(1200 + 600*np.sin((m.month/12)*2*np.pi) + rng.integers(0,200))
        mql = int(leads * (0.45 + 0.1*rng.random()))
        sql = int(mql * (0.55 + 0.1*rng.random()))
        demos = int(sql * (0.62 + 0.08*rng.random()))
        wins = int(demos * (0.34 + 0.07*rng.random()))
        funnel.append({"month": m, "leads": leads, "mql": mql, "sql": sql, "demos": demos, "wins": wins})
    funnel = pd.DataFrame(funnel)

    # Chargebacks monthly (subset of fraud)
    if not txns.empty:
        chargebacks = (
            txns.assign(month=lambda x: x["date"].values.astype("datetime64[M]"))
                .groupby("month", as_index=False)[["fraud_loss"]].sum()
        )
        chargebacks["chargebacks"] = (chargebacks["fraud_loss"] * 0.45).round(2)
    else:
        chargebacks = pd.DataFrame(columns=["month","fraud_loss","chargebacks"])

    return cust, subs, txns, funnel, chargebacks, regions, products, plans, reps, segments, industries, days.min(), days.max()

# -----------------------------
# Data
# -----------------------------
cust, subs, txns, funnel, chargebacks, REGIONS, PRODUCTS, PLANS, REPS, SEGMENTS, INDUSTRIES, MIN_DATE, MAX_DATE = make_data()

# Precompute helpers
if not txns.empty:
    txns["month"] = txns["date"].values.astype("datetime64[M]")
subs_summary = subs.groupby("month", as_index=False).agg(
    active_accounts=("active","sum"),
    mrr=("mrr","sum")
) if not subs.empty else pd.DataFrame(columns=["month","active_accounts","mrr"])

revenue_monthly = (txns.groupby("month", as_index=False).agg(
    gross=("gross_revenue","sum"),
    net=("net_revenue","sum"),
    refunds=("refunds","sum"),
    fraud=("fraud_loss","sum"),
    txn_count=("txn_count","sum")
) if not txns.empty else pd.DataFrame(columns=["month","gross","net","refunds","fraud","txn_count"]))

# -----------------------------
# Sidebar: Global filters
# -----------------------------
st.sidebar.title("Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=(MAX_DATE - relativedelta(months=6), MAX_DATE),
    min_value=MIN_DATE,
    max_value=MAX_DATE
)
region_sel = st.sidebar.multiselect("Region", REGIONS, default=REGIONS)
product_sel = st.sidebar.multiselect("Product", PRODUCTS, default=PRODUCTS)
plan_sel = st.sidebar.multiselect("Plan", PLANS, default=PLANS)
segment_sel = st.sidebar.multiselect("Segment", SEGMENTS, default=SEGMENTS)

# Apply filters
mask_cust = (
    cust["region"].isin(region_sel) &
    cust["plan"].isin(plan_sel) &
    cust["segment"].isin(segment_sel) &
    (cust["acquired_date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
)
cust_f = cust.loc[mask_cust]

if not txns.empty:
    mask_txn = (
        txns["region"].isin(region_sel) &
        txns["product_primary"].isin(product_sel) &
        txns["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
    )
    txns_f = txns.loc[mask_txn]
else:
    txns_f = txns.copy()

mask_subs = (
    subs["region"].isin(region_sel) &
    subs["plan"].isin(plan_sel) &
    subs["segment"].isin(segment_sel) &
    subs["month"].between(
        pd.to_datetime(pd.to_datetime(date_range[0]).replace(day=1)),
        pd.to_datetime(pd.to_datetime(date_range[1]).replace(day=1))
    )
)
subs_f = subs.loc[mask_subs]

# KPI helpers
def kpi_delta(curr, prev):
    if prev == 0:
        return 0.0
    return ((curr - prev) / prev) * 100

def month_agg(df):
    if df.empty:
        return df
    return df.groupby("month", as_index=False).sum(numeric_only=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("### 💳 Fintech Performance Cockpit")
st.caption("Single source of truth for Sales, Operations, Risk, Finance, Customer success, and Board overview. Fictional data for portfolio demonstration.")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_sales, tab_ops, tab_risk, tab_finance, tab_customer, tab_board = st.tabs(
    ["Overview", "Sales", "Operations", "Risk & Fraud", "Finance & Unit Economics", "Customer", "Board Pack"]
)

# ========== OVERVIEW ==========
with tab_overview:
    st.subheader("Company Pulse")

    m_rev_all = month_agg(txns.groupby("month").agg(net=("net_revenue","sum")).reset_index()) if not txns.empty else pd.DataFrame(columns=["month","net"])
    m_rev = txns_f.groupby("month", as_index=False).agg(net=("net_revenue","sum")) if not txns_f.empty else pd.DataFrame(columns=["month","net"])
    m_accounts = subs_f.groupby("month", as_index=False).agg(active=("active","sum"), mrr=("mrr","sum")) if not subs_f.empty else pd.DataFrame(columns=["month","active","mrr"])

    if not m_rev.empty:
        cur_m = m_rev["month"].max()
        prev_m = cur_m - relativedelta(months=1)
        cur_rev = float(m_rev.loc[m_rev["month"] == cur_m, "net"].sum())
        prev_rev = float(m_rev.loc[m_rev["month"] == prev_m, "net"].sum()) if (m_rev["month"]==prev_m).any() else 0.0

        cur_accts = int(m_accounts.loc[m_accounts["month"] == cur_m, "active"].sum()) if (m_accounts["month"]==cur_m).any() else 0
        prev_accts = int(m_accounts.loc[m_accounts["month"] == prev_m, "active"].sum()) if (m_accounts["month"]==prev_m).any() else 0

        cur_mrr = float(m_accounts.loc[m_accounts["month"] == cur_m, "mrr"].sum()) if (m_accounts["month"]==cur_m).any() else 0.0
        prev_mrr = float(m_accounts.loc[m_accounts["month"] == prev_m, "mrr"].sum()) if (m_accounts["month"]==prev_m).any() else 0.0
    else:
        cur_rev=prev_rev=cur_mrr=prev_mrr=0.0
        cur_accts=prev_accts=0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monthly Net Revenue", f"${cur_rev:,.0f}", f"{kpi_delta(cur_rev, prev_rev):.1f}% vs prev")
    c2.metric("Active Accounts", f"{cur_accts:,}", f"{kpi_delta(cur_accts, prev_accts):.1f}% vs prev")
    c3.metric("MRR", f"${cur_mrr:,.0f}", f"{kpi_delta(cur_mrr, prev_mrr):.1f}% vs prev")
    c4.metric("Gross Margin Proxy", "≈ 92.5%", "+0.3 pp vs prev")

    if not m_rev_all.empty:
        fig = px.area(m_rev_all, x="month", y="net", title="Company Net Revenue Trend (all regions, all products)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No revenue data available for the current filters.")

    col1, col2 = st.columns(2)
    if not txns_f.empty:
        rev_by_region = txns_f.groupby("region", as_index=False)["net_revenue"].sum().sort_values("net_revenue", ascending=False)
        col1.plotly_chart(px.bar(rev_by_region, x="region", y="net_revenue", title="Revenue by Region"), use_container_width=True)

        rev_by_product = txns_f.groupby("product_primary", as_index=False)["net_revenue"].sum().sort_values("net_revenue", ascending=False)
        col2.plotly_chart(px.pie(rev_by_product, names="product_primary", values="net_revenue", title="Revenue mix by Product"), use_container_width=True)
    else:
        col1.info("No transaction data in selection.")
        col2.info("No transaction data in selection.")

# ========== SALES ==========
with tab_sales:
    st.subheader("Sales Performance")

    fsel = funnel[funnel["month"].between(
        pd.to_datetime(pd.to_datetime(date_range[0]).replace(day=1)),
        pd.to_datetime(pd.to_datetime(date_range[1]).replace(day=1))
    )]
    if not fsel.empty:
        fsel = fsel.copy()
        fsel["win_rate"] = (fsel["wins"] / fsel["demos"].replace(0, np.nan)) * 100
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Leads (avg)", f"{int(fsel['leads'].mean()):,}")
        c2.metric("MQL (avg)", f"{int(fsel['mql'].mean()):,}")
        c3.metric("SQL (avg)", f"{int(fsel['sql'].mean()):,}")
        c4.metric("Demos (avg)", f"{int(fsel['demos'].mean()):,}")
        c5.metric("Win rate (avg)", f"{fsel['win_rate'].mean():.1f}%")

        col1, col2 = st.columns(2)
        col1.plotly_chart(px.line(fsel, x="month", y=["leads","mql","sql","demos","wins"], title="Funnel over time"), use_container_width=True)
    else:
        st.info("No funnel data for the selected period.")
        col1, col2 = st.columns(2)

    # Rep leaderboard (safe even if empty)
    if not txns_f.empty and "sales_rep" in txns_f.columns:
        rep_rev = txns_f.groupby("sales_rep", as_index=False)["net_revenue"].sum().sort_values("net_revenue", ascending=False).head(20)
        col2.plotly_chart(px.bar(rep_rev, x="sales_rep", y="net_revenue", title="Rep leaderboard (Net Revenue)"), use_container_width=True)
    else:
        col2.info("No rep revenue available in selection.")

    # Region x product heatmap
    col3, col4 = st.columns(2)
    if not txns_f.empty:
        heat = txns_f.groupby(["region","product_primary"], as_index=False)["net_revenue"].sum()
        pt = heat.pivot(index="region", columns="product_primary", values="net_revenue").fillna(0)
        col3.plotly_chart(px.imshow(pt, text_auto=".2s", aspect="auto", title="Revenue Heatmap: Region x Product"), use_container_width=True)

        # Simple scatter (no statsmodels required)
        growth = subs_f.groupby("month", as_index=False)["active"].sum() if not subs_f.empty else pd.DataFrame(columns=["month","active"])
        proxy = (fsel.merge(growth, on="month", how="left")) if not fsel.empty else pd.DataFrame(columns=["month","wins","active"])
        if not proxy.empty:
            col4.plotly_chart(px.scatter(proxy, x="wins", y="active", title="Wins vs Active Accounts (proxy)"), use_container_width=True)
        else:
            col4.info("Not enough data for wins vs active accounts.")
    else:
        col3.info("No revenue by region/product.")
        col4.info("Not enough data for scatter.")

# ========== OPERATIONS ==========
with tab_ops:
    st.subheader("Operational Health")
    if not txns_f.empty:
        daily = txns_f.groupby("date", as_index=False).agg(
            txn=("txn_count","sum"),
            refunds=("refunds","sum"),
            fraud=("fraud_loss","sum"),
            net=("net_revenue","sum"),
            gross=("gross_revenue","sum")
        )
        daily["success_rate"] = np.where(daily["txn"]>0, 100*(1 - (daily["refunds"]+daily["fraud"])/daily["gross"].replace(0,np.nan)), np.nan)

        c1, c2, c3 = st.columns(3)
        c1.metric("Avg daily transactions", f"{int(daily['txn'].mean()):,}")
        c2.metric("Transaction success rate", f"{daily['success_rate'].mean():.2f}%")
        c3.metric("Refunds as % of gross", f"{100*daily['refunds'].sum()/daily['gross'].sum():.2f}%")

        st.plotly_chart(px.line(daily, x="date", y=["txn","success_rate"], title="Daily volume and success rate"), use_container_width=True)

        col1, col2 = st.columns(2)
        by_seg = txns_f.groupby("segment", as_index=False)[["txn_count","gross_revenue","refunds"]].sum()
        by_seg["refund_rate_pct"] = 100 * by_seg["refunds"] / by_seg["gross_revenue"].replace(0, np.nan)
        col1.plotly_chart(px.bar(by_seg, x="segment", y="refund_rate_pct", title="Refund rate by Segment"), use_container_width=True)

        by_region_day = txns_f.groupby(["region","date"], as_index=False)["txn_count"].sum()
        col2.plotly_chart(px.line(by_region_day, x="date", y="txn_count", color="region", title="Daily volume by region"), use_container_width=True)
    else:
        st.info("No operational data available for the current filters.")

# ========== RISK & FRAUD ==========
with tab_risk:
    st.subheader("Risk and Fraud")
    if not txns_f.empty:
        fr = txns_f.groupby("month", as_index=False).agg(fraud=("fraud_loss","sum"), gross=("gross_revenue","sum"))
        fr["fraud_rate_pct"] = 100 * fr["fraud"] / fr["gross"].replace(0, np.nan)
        c1, c2 = st.columns(2)
        c1.metric("Fraud rate (avg)", f"{fr['fraud_rate_pct'].mean():.3f}%")
        c2.metric("Fraud loss (sum)", f"${fr['fraud'].sum():,.0f}")

        st.plotly_chart(px.bar(fr, x="month", y="fraud_rate_pct", title="Monthly fraud rate %"), use_container_width=True)

        daily_fraud = txns_f.groupby("date", as_index=False)["fraud_loss"].sum()
        thr = daily_fraud["fraud_loss"].mean() + 3*daily_fraud["fraud_loss"].std()
        alerts = daily_fraud[daily_fraud["fraud_loss"] > thr]

        col1, col2 = st.columns(2)
        fig_df = px.line(daily_fraud, x="date", y="fraud_loss", title="Daily fraud loss (3-sigma threshold shown)")
        fig_df.add_hline(y=float(thr))
        col1.plotly_chart(fig_df, use_container_width=True)
        col2.dataframe(alerts.rename(columns={"fraud_loss":"fraud_loss_usd"}), use_container_width=True, height=320)

        cb = chargebacks.merge(fr[["month","fraud_rate_pct"]], on="month", how="left")
        st.plotly_chart(px.line(cb, x="month", y=["chargebacks","fraud_rate_pct"], title="Chargebacks and Fraud rate"), use_container_width=True)
    else:
        st.info("No fraud data for the current filters.")

# ========== FINANCE & UNIT ECONOMICS ==========
with tab_finance:
    st.subheader("Finance and Unit Economics")
    CAC_PER_WIN = 140.0

    if not subs_f.empty:
        arpu = subs_f.groupby("month", as_index=False)["mrr"].sum().merge(
            subs_f.groupby("month", as_index=False)["active"].sum(), on="month", how="left"
        )
        arpu["ARPU"] = arpu["mrr"] / arpu["active"].replace(0, np.nan)
        arpu["CAC"] = CAC_PER_WIN

        act_by_month = subs_f.groupby("month", as_index=False)["active"].sum().rename(columns={"active":"active_accts"})
        act_by_month["churned"] = act_by_month["active_accts"].diff(-1) * -1
        act_by_month["churn_rate_pct"] = 100 * (act_by_month["churned"].clip(lower=0) / act_by_month["active_accts"].replace(0, np.nan))
        churn_rate = act_by_month["churn_rate_pct"].dropna().mean()
        mean_arpu = arpu["ARPU"].dropna().mean() if not arpu["ARPU"].dropna().empty else 0
        ltv = mean_arpu / (churn_rate/100) if churn_rate and churn_rate > 0 else np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("Average ARPU", f"${mean_arpu:,.2f}")
        c2.metric("Avg churn rate", f"{(churn_rate if churn_rate else 0):.2f}%")
        c3.metric("LTV (simple)", f"${ltv:,.0f}" if np.isfinite(ltv) else "n/a")

        col1, col2 = st.columns(2)
        col1.plotly_chart(px.line(arpu, x="month", y="ARPU", title="ARPU over time"), use_container_width=True)
        col2.plotly_chart(px.line(act_by_month, x="month", y="churn_rate_pct", title="Churn rate over time"), use_container_width=True)
    else:
        st.info("Not enough subscription data to compute ARPU/LTV.")
        arpu = pd.DataFrame()
        act_by_month = pd.DataFrame()

    # P&L lite: net revenue minus CAC proxy
    if not txns.empty and not funnel.empty:
        wins_series = funnel.set_index("month")["wins"].reindex(txns["month"].sort_values().unique()).fillna(method="ffill").fillna(0)
        pnl = pd.DataFrame({
            "month": wins_series.index,
            "net_revenue": txns.groupby("month")["net_revenue"].sum().reindex(wins_series.index).values,
        })
        pnl["cac_spend"] = wins_series.values * CAC_PER_WIN
        pnl["contribution"] = pnl["net_revenue"] - pnl["cac_spend"]
        st.plotly_chart(px.bar(pnl, x="month", y=["net_revenue","cac_spend","contribution"], title="Contribution after CAC proxy"), use_container_width=True)
    else:
        st.info("Not enough data for contribution view.")

# ========== CUSTOMER ==========
with tab_customer:
    st.subheader("Customer Insights")

    if not cust.empty and not subs.empty:
        cohort = cust.copy()
        cohort["cohort_month"] = cohort["acquired_date"].values.astype("datetime64[M]")
        active_map = subs[["customer_id","month","active"]]
        first_month = cohort[["customer_id","cohort_month"]]
        merged = active_map.merge(first_month, on="customer_id", how="left").dropna(subset=["cohort_month"])
        merged["period_index"] = ((merged["month"].dt.to_period("M") - pd.to_datetime(merged["cohort_month"]).dt.to_period("M")).apply(lambda x: x.n)).astype(int)
        cohort_pivot = merged.groupby(["cohort_month","period_index"], as_index=False)["active"].mean()
        coh = cohort_pivot[cohort_pivot["period_index"].between(0, 12)]
        heat = coh.pivot(index="cohort_month", columns="period_index", values="active").fillna(0)

        col1, col2 = st.columns(2)
        col1.plotly_chart(px.imshow(heat, aspect="auto", title="Retention Cohort (share active) 0-12 months", text_auto=".2f"), use_container_width=True)

        seg = cust_f.groupby(["segment","plan"], as_index=False)["customer_id"].count().rename(columns={"customer_id":"count"}) if not cust_f.empty else pd.DataFrame(columns=["segment","plan","count"])
        if not seg.empty:
            col2.plotly_chart(px.treemap(seg, path=["segment","plan"], values="count", title="Segment x Plan distribution"), use_container_width=True)
        else:
            col2.info("No customers in current filter for segmentation view.")

        if not txns_f.empty:
            topc = txns_f.groupby("customer_id", as_index=False)["net_revenue"].sum().sort_values("net_revenue", ascending=False).head(50)
            topc = topc.merge(cust[["customer_id","region","industry","segment","sales_rep"]], on="customer_id", how="left")
            st.markdown("**Top customers by net revenue**")
            st.dataframe(topc, use_container_width=True, height=320)
        else:
            st.info("No customer revenue under current filters.")
    else:
        st.info("Customer/subscription data not available.")

# ========== BOARD PACK ==========
with tab_board:
    st.subheader("Board Pack Summary")
    if not txns.empty and not subs.empty:
        last_month = txns["month"].max()
        prev_month = last_month - relativedelta(months=1)
        rm = txns[txns["month"] == last_month]["net_revenue"].sum()
        rp = txns[txns["month"] == prev_month]["net_revenue"].sum()
        growth = kpi_delta(rm, rp)
        act = subs[subs["month"] == last_month]["active"].sum()
        mrr_ = subs[subs["month"] == last_month]["mrr"].sum()
        fraud_rate = (txns[txns["month"] == last_month]["fraud_loss"].sum() / txns[txns["month"] == last_month]["gross_revenue"].sum()) * 100 if txns[txns["month"] == last_month]["gross_revenue"].sum() else 0
    else:
        rm=rp=growth=act=mrr_=fraud_rate=0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Revenue (last month)", f"${rm:,.0f}", f"{growth:.1f}% MoM")
    c2.metric("Active Accounts", f"{act:,}")
    c3.metric("MRR", f"${mrr_:,.0f}")
    c4.metric("Fraud rate", f"{fraud_rate:.2f}%")

    col1, col2 = st.columns(2)
    if not txns.empty:
        board_rev = txns.groupby("month", as_index=False)["net_revenue"].sum()
        col1.plotly_chart(px.line(board_rev, x="month", y="net_revenue", title="Net revenue trend"), use_container_width=True)

        rshare = txns.groupby("region", as_index=False)["net_revenue"].sum().sort_values("net_revenue", ascending=False)
        col2.plotly_chart(px.pie(rshare, names="region", values="net_revenue", title="Region share of revenue"), use_container_width=True)
    else:
        col1.info("No revenue trend available.")
        col2.info("No region share available.")

    st.markdown("""
**Narrative highlights**
- Growth driven by POS and Wallet in Senegal and Nigeria
- Churn improving in Mid Market and Enterprise segments
- Fraud rate within tolerance but weekend spikes require continued monitoring

**Next actions**
1. Double down on POS in high growth districts with targeted offers
2. Expand Customer Success playbooks for Starter to Growth upgrades
3. Tighten weekend risk rules and velocity checks on Paylink
4. Shift paid spend toward channels that convert to demos with higher win rates
""")

# Footer
st.caption("Fictional data. Built with Streamlit, Plotly, and Pandas to demonstrate a data-driven decision dashboard for a fintech context.")
