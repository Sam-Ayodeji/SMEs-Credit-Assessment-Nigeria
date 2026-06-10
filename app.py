"""
SME Credit Research - Streamlit Web App
World Bank Enterprise Survey Nigeria 2025
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr
import io
import datetime

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMEs Credit Uptake, Access, Constraints, and Creditworthiness Assessment in Nigeria",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 16px 18px; border: 1px solid #e9ecef;
    }
    .metric-card .label { font-size: 12px; color: #6c757d; margin-bottom: 4px; }
    .metric-card .value { font-size: 24px; font-weight: 600; color: #1a1a2e; }
    .metric-card .sub   { font-size: 11px; color: #6c757d; margin-top: 3px; }
    .section-note {
        background: #e8f4fd; border-left: 4px solid #1a73e8;
        padding: 10px 14px; border-radius: 0 6px 6px 0;
        font-size: 13px; color: #1a1a2e; margin-bottom: 12px;
    }
    .finding-box {
        background: #f0faf4; border-left: 4px solid #1a6b3c;
        padding: 10px 14px; border-radius: 0 6px 6px 0;
        font-size: 13px; color: #1a1a2e; margin-top: 8px; margin-bottom: 4px;
    }
    .finding-box b { color: #1a6b3c; }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.0rem !important; }
</style>
""", unsafe_allow_html=True)

# ─── Colour palette ───────────────────────────────────────────────────────────
COLORS = {
    "green"  : "#1a6b3c",
    "red"    : "#c0392b",
    "blue"   : "#2980b9",
    "purple" : "#8e44ad",
    "orange" : "#e67e22",
    "teal"   : "#16a085",
    "dark"   : "#2c3e50",
}
ZONE_COLORS = ["#1a6b3c", "#c0392b", "#2980b9", "#8e44ad", "#e67e22", "#16a085"]
ZONES = ["North Central", "North East", "North West", "South East", "South South", "South West"]

# ─── Zone map centroids (lat/lon for Nigeria geo-political zones) ──────────────
ZONE_COORDS = {
    "North Central": (9.0,  7.5),
    "North East":    (11.5, 13.0),
    "North West":    (11.5, 7.0),
    "South East":    (5.8,  7.5),
    "South South":   (5.0,  6.0),
    "South West":    (7.0,  3.9),
}

# ─── Load real WBES Nigeria cleaned dataset ───────────────────────────────────
@st.cache_data
def load_data():
    """Load and post-process the notebook-exported cleaned CSV."""
    df = pd.read_csv("wbes_nigeria_cleaned.csv")

    # sme_size: PyArrow-backed StringDtype → plain Python str
    df["sme_size"] = df["sme_size"].fillna("Medium").astype(str)

    # creditworthy: int64 → float for consistency
    df["creditworthy"] = df["creditworthy"].astype(float)

    float_cols = [
        "k30", "c30a", "j30f", "j30e", "e30", "d30a", "j30a", "j30b",
        "k17", "m1a", "k20a1",
        "b4", "b7a", "k21", "k6", "c22b", "l10", "h1", "h5", "h8",
        "credit_uptake", "has_credit", "loan_approved",
        "k3a", "k3bc", "k3e", "k3f", "k3dgh",
        "k5a", "k5bc", "k5e", "k5f",
        "k14a", "k14b", "k14c", "k14d",
        "k33", "k38", "f1", "sales_growth", "firm_age", "b7",
        "ln_sales", "ln_employees", "l1",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data()

# ─── Helper functions ─────────────────────────────────────────────────────────
def sig_star(p):
    return "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))

def pct(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return round(s.mean() * 100, 1) if len(s) > 0 else 0.0

def finding(text):
    """Render a green key-finding callout box."""
    st.markdown(f'<div class="finding-box">💡 <b>Key finding:</b> {text}</div>',
                unsafe_allow_html=True)

def dynamic_finding(fn, *args, **kwargs):
    """Call fn(*args) to produce finding text from filtered data and render it."""
    try:
        finding(fn(*args, **kwargs))
    except Exception:
        pass

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### SME Credit Research")
    st.markdown("World Bank Enterprise Survey  \nNigeria 2025  |  N = 1,043 firms")
    st.divider()

    st.markdown("**Filters**")
    zone_filter = st.multiselect(
        "Geo-political zone",
        options=ZONES,
        default=ZONES,
        help="Filter all charts and tables by zone"
    )
    sector_filter = st.multiselect(
        "Sector",
        options=sorted(df["sector_label"].unique()),
        default=sorted(df["sector_label"].unique())
    )
    size_options = [s for s in ["Small", "Medium", "Large"] if s in df["sme_size"].values]
    size_filter = st.multiselect(
        "Firm size",
        options=size_options,
        default=size_options
    )
    st.divider()
    st.markdown("**Navigate**")
    page = st.radio(
        "",
        options=[
            "Overview",
            "Obj 1: Credit Uptake",
            "Obj 2: Credit Access",
            "Obj 3: Financing Structure",
            "Obj 4: Barriers",
            "Obj 5 & 6: Transparency & Gender",
            "Obj 7: Firm Performance",
            "Obj 8 & 9: Creditworthiness",
            "Zone Comparison",
            "🗺️ Nigeria Zone Map",
            "📊 Obstacle Intelligence",
            "📄 Executive Report",
        ],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Use the filters to explore zone, sector, and size subgroups dynamically.")

# ─── Filtered data ────────────────────────────────────────────────────────────
mask = (
    df["region_label"].isin(zone_filter) &
    df["sector_label"].isin(sector_filter) &
    df["sme_size"].isin(size_filter)
)
dff = df[mask].copy()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":

    st.markdown("""
    <div style="text-align:center; padding-bottom:20px;">
        <h1>SMEs Credit Assessment in Nigeria</h1>
        <h3>By Insight Analytics Consult</h3>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        f"World Bank Enterprise Survey | N = {len(df):,} firms "
        f"| Survey Year: 2025"
    )
    st.markdown("""
<div class="section-note">
This dashboard presents empirical findings from the World Bank Enterprise Survey Nigeria 2025
across nine research objectives. Use the sidebar filters to explore national vs zone-level patterns.
All charts are interactive - hover for values, click legends to toggle series.
</div>
""", unsafe_allow_html=True)

    cols = st.columns(4)
    kpis = [
        ("Firms surveyed",     f"{len(dff):,}",                               "In filtered selection"),
        ("Applied for credit", f"{pct(dff['credit_uptake'])}%",               "Credit demand rate"),
        ("Have a bank loan",   f"{pct(dff['has_credit'])}%",                  "Existing credit stock"),
        ("Approval rate",      f"{pct(dff['loan_approved'].dropna())}%",      "Among applicants"),
    ]
    for col, (lbl, val, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
<div class="metric-card">
  <div class="label">{lbl}</div>
  <div class="value">{val}</div>
  <div class="sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    kpis2 = [
        ("Externally audited", f"{pct(dff['k21'])}%",           "Financial transparency"),
        ("Internal WC funding", f"{dff['k3a'].mean():.1f}%",    "Self-financing dominance"),
        ("Bank WC funding",     f"{dff['k3bc'].mean():.1f}%",   "Formal bank reliance"),
        ("Female-owned firms",  f"{pct(dff['b4'])}%",           "Any female ownership"),
    ]
    for col, (lbl, val, sub) in zip(cols2, kpis2):
        with col:
            st.markdown(f"""
<div class="metric-card">
  <div class="label">{lbl}</div>
  <div class="value">{val}</div>
  <div class="sub">{sub}</div>
</div>""", unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Credit uptake by zone")
        zone_u = dff.groupby("region_label")["credit_uptake"].mean().mul(100).reset_index()
        zone_u.columns = ["Zone", "Rate (%)"]
        zone_u = zone_u.sort_values("Rate (%)", ascending=True)
        fig = px.bar(zone_u, x="Rate (%)", y="Zone", orientation="h",
                     color="Rate (%)", color_continuous_scale="Greens",
                     text="Rate (%)", template="plotly_white")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="", xaxis_title="% Applied")
        st.plotly_chart(fig, use_container_width=True)
        best_zone = zone_u.iloc[-1]["Zone"]
        best_rate = zone_u.iloc[-1]["Rate (%)"]
        worst_zone = zone_u.iloc[0]["Zone"]
        worst_rate = zone_u.iloc[0]["Rate (%)"]
        finding(f"{best_zone} leads credit uptake at {best_rate:.1f}%, while {worst_zone} trails "
                f"at {worst_rate:.1f}% - a {best_rate - worst_rate:.1f} pp gap reflecting deep "
                f"regional divergence in formal credit demand.")

    with c2:
        st.subheader("Financing structure - working capital")
        wc_labels = {"k3a": "Internal funds", "k3bc": "Bank borrowing",
                     "k3e": "Non-bank FI", "k3f": "Supplier credit", "k3dgh": "Informal/other"}
        wc_vals = [dff[k].mean() for k in wc_labels]
        fig = px.pie(names=list(wc_labels.values()), values=wc_vals,
                     hole=0.4, template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(textinfo="percent+label", textfont_size=11)
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        finding(f"Internal funds account for {dff['k3a'].mean():.1f}% of working capital; "
                f"bank borrowing is negligible at {dff['k3bc'].mean():.1f}%. Nigerian SMEs "
                f"operate in a near-credit-free equilibrium sustained almost entirely by retained earnings.")

    st.subheader("Research objectives summary")
    obj_df = pd.DataFrame({
        "Objective": [
            "Obj 1: Credit Uptake", "Obj 2: Credit Access", "Obj 3: Financing Structure",
            "Obj 4: Barriers", "Obj 5: Transparency", "Obj 6: Gender",
            "Obj 7: Firm Performance", "Obj 8: Creditworthiness", "Obj 9: ML Features",
        ],
        "Key finding": [
            f"Only {pct(dff['credit_uptake'].dropna()):.1f}% of Nigerian SMEs applied for formal credit. Product innovation, R&D activity, and website ownership are the only significant uptake predictors.",
            f"~{pct(dff['loan_approved'].dropna()):.0f}% approval rate among applicants. Website ownership is the sole binary approval predictor; firm size and sales are the key continuous predictors.",
            f"Internal funds cover {dff['k3a'].mean():.0f}% of working capital; bank financing only {dff['k3bc'].mean():.1f}%. Self-financing is the norm across all zones.",
            "Electricity (mean 2.57/4) and access to finance (2.09/4) are the two most severe obstacles, together cited as the biggest obstacle by 58.7% of firms.",
            "Business website ownership predicts all three credit outcomes, including a 26.1 pp loan approval advantage. External audit significantly predicts credit ownership (+6.1 pp).",
            "No statistically significant national gender gap. Zone-level analysis reveals concentrated female disadvantage in North West and South East (gaps up to −13.3 pp).",
            "Performance does not predict credit demand, but strongly predicts credit ownership. Number of employees (p<0.001) and annual sales (p=0.002) are the significant predictors.",
            "Six high-priority ML features: number of employees, annual sales, e-payments made, website presence, external audit, and (endogenous) WC bank funding share.",
            "Three staged ML models proposed: creditworthiness prediction, loan approval, and loan amount recommendation - all grounded in WBES-derived features.",
        ]
    })
    st.dataframe(obj_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OBJ 1 - CREDIT UPTAKE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Obj 1: Credit Uptake":
    st.header("Objective 1: Determinants of SME Credit Uptake")
    st.markdown("""
<div class="section-note">
Research question: What factors influence SMEs' decisions to apply for formal credit?<br>
Dependent variable: <code>credit_uptake</code> - did the firm apply for a loan in the last fiscal year?
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["National charts", "Zone breakdown", "Statistical tests"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Application rate by sector")
            sec_u = (dff.groupby("sector_label")["credit_uptake"]
                     .agg(N="count", Applied=lambda x: x.sum(),
                          Rate=lambda x: round(x.mean() * 100, 1))
                     .reset_index().sort_values("Rate", ascending=True))
            fig = px.bar(sec_u, x="Rate", y="sector_label", orientation="h",
                         color="Rate", color_continuous_scale="Blues",
                         text="Rate", template="plotly_white",
                         labels={"sector_label": "Sector", "Rate": "% Applied"})
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(height=320, margin=dict(l=10, r=30, t=10, b=10),
                              yaxis_title="", xaxis_range=[0, 55])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Table 1 - Application rate by sector**")
            sec_u.columns = ["Sector", "N firms", "Applied N", "Application rate (%)"]
            st.dataframe(sec_u.sort_values("Application rate (%)", ascending=False),
                         use_container_width=True, hide_index=True)
            if not sec_u.empty:
                top_sec = sec_u.iloc[-1]
                bot_sec = sec_u.iloc[0]
                finding(f"{top_sec['Sector']} records the highest application rate "
                        f"({top_sec['Application rate (%)']:.1f}%), while {bot_sec['Sector']} "
                        f"is lowest ({bot_sec['Application rate (%)']:.1f}%). Capital intensity "
                        f"and growth orientation in manufacturing drive higher credit demand.")

        with c2:
            st.subheader("Application rate by firm size")
            size_u = (dff.groupby("sme_size")["credit_uptake"]
                      .agg(N="count", Applied=lambda x: x.sum(),
                           Rate=lambda x: round(x.mean() * 100, 1))
                      .reset_index().sort_values("Rate", ascending=False))
            fig = px.bar(size_u, x="sme_size", y="Rate",
                         color="sme_size", text="Rate",
                         color_discrete_sequence=[COLORS["orange"], COLORS["blue"], COLORS["green"]],
                         template="plotly_white",
                         labels={"sme_size": "Firm size", "Rate": "% Applied"})
            fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            fig.update_layout(height=320, showlegend=False,
                              margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 60])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Table 2 - Application rate by firm size**")
            size_u.columns = ["Firm size", "N firms", "Applied N", "Application rate (%)"]
            st.dataframe(size_u, use_container_width=True, hide_index=True)
            if not size_u.empty:
                rates = size_u.set_index("Firm size")["Application rate (%)"].to_dict()
                finding(f"Large firms apply at {rates.get('Large', 0):.1f}% versus "
                        f"{rates.get('Small', 0):.1f}% for small firms. Medium firms "
                        f"({rates.get('Medium', 0):.1f}%) show a 'missing middle' pattern - "
                        f"outgrown micro-credit but lacking documentation for bank loans.")

        st.subheader("Credit uptake rate by formalisation indicator")
        enablers = {"Bank account": "k6", "External audit": "k21",
                    "Has website": "c22b", "Formal training": "l10"}
        enab_rows = []
        for lbl, col in enablers.items():
            r_without = pct(dff[dff[col] == 0]["credit_uptake"].dropna())
            r_with    = pct(dff[dff[col] == 1]["credit_uptake"].dropna())
            enab_rows.append({"Indicator": lbl, "Without (%)": r_without, "With (%)": r_with,
                               "Difference (pp)": round(r_with - r_without, 1)})
        enab_df = pd.DataFrame(enab_rows)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Without indicator", x=enab_df["Indicator"],
                             y=enab_df["Without (%)"], marker_color=COLORS["red"],
                             text=enab_df["Without (%)"].map(lambda v: f"{v:.1f}%"),
                             textposition="outside"))
        fig.add_trace(go.Bar(name="With indicator", x=enab_df["Indicator"],
                             y=enab_df["With (%)"], marker_color=COLORS["green"],
                             text=enab_df["With (%)"].map(lambda v: f"{v:.1f}%"),
                             textposition="outside"))
        fig.update_layout(barmode="group", height=350, template="plotly_white",
                          yaxis_range=[0, 55], yaxis_title="% Applied",
                          margin=dict(l=10, r=10, t=10, b=10), legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Table 3 - Credit uptake by formalisation indicators**")
        st.dataframe(enab_df, use_container_width=True, hide_index=True)
        if not enab_df.empty:
            best_row = enab_df.loc[enab_df["Difference (pp)"].idxmax()]
            finding(f"'{best_row['Indicator']}' shows the largest uptake advantage: firms with it "
                    f"apply at {best_row['With (%)']:.1f}% vs {best_row['Without (%)']:.1f}% without "
                    f"(+{best_row['Difference (pp)']:.1f} pp). These formalisation markers reduce "
                    f"perceived rejection risk and lower the cost of applying.")

    with tab2:
        st.subheader("Credit uptake by geo-political zone × sector")
        zone_u2 = (dff.groupby(["region_label", "sector_label"])["credit_uptake"]
                   .mean().mul(100).round(1).reset_index())
        zone_u2.columns = ["Zone", "Sector", "Application rate (%)"]
        fig = px.bar(zone_u2, x="Sector", y="Application rate (%)",
                     color="Zone", barmode="group", template="plotly_white",
                     color_discrete_sequence=ZONE_COLORS,
                     labels={"Application rate (%)": "% Applied"})
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=40),
                          legend_orientation="h", xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Zone profile table")
        zone_profile = []
        for zone in ZONES:
            sub = dff[dff["region_label"] == zone]
            if sub.shape[0] < 3:
                continue
            zone_profile.append({
                "Zone": zone,
                "N firms": sub.shape[0],
                "Applied N": int(sub["credit_uptake"].sum()),
                "Application rate (%)": pct(sub["credit_uptake"]),
                "Uptake - audited firms (%)": pct(sub[sub["k21"] == 1]["credit_uptake"].dropna()),
                "% small firms": round((sub["sme_size"] == "Small").mean() * 100, 1),
                "% female-owned": pct(sub["b4"].dropna()),
            })
        zone_prof_df = pd.DataFrame(zone_profile)
        st.markdown("**Table 4 - Zone-level credit uptake profile**")
        st.dataframe(zone_prof_df.sort_values("Application rate (%)", ascending=False),
                     use_container_width=True, hide_index=True)
        if not zone_prof_df.empty:
            high = zone_prof_df.loc[zone_prof_df["Uptake - audited firms (%)"].idxmax()]
            finding(f"In {high['Zone']}, audited firms apply at {high['Uptake - audited firms (%)']:.1f}% "
                    f"vs an overall zone rate of {high['Application rate (%)']:.1f}% - suggesting that "
                    f"external audit functions as a particularly strong credit-demand enabler in this zone.")

    with tab3:
        st.subheader("Chi-square significance tests - binary predictors vs credit uptake")
        predictors = {"External audit": "k21", "Bank account": "k6", "Has website": "c22b",
                      "Female ownership": "b4", "Female manager": "b7a",
                      "Formal training": "l10", "Product innovation": "h1",
                      "Process innovation": "h5", "R&D spending": "h8"}
        rows = []
        for lbl, col in predictors.items():
            sub = dff[[col, "credit_uptake"]].dropna()
            if sub.shape[0] < 10:
                continue
            ct = pd.crosstab(sub[col], sub["credit_uptake"])
            if ct.shape != (2, 2):
                continue
            chi2, p, _, _ = chi2_contingency(ct)
            r0 = round(ct.loc[0, 1] / ct.loc[0].sum() * 100, 1) if 0 in ct.index else np.nan
            r1 = round(ct.loc[1, 1] / ct.loc[1].sum() * 100, 1) if 1 in ct.index else np.nan
            rows.append({"Predictor": lbl, "Rate without (%)": r0, "Rate with (%)": r1,
                         "Difference (pp)": round(r1 - r0, 1),
                         "Chi²": round(chi2, 3), "p-value": round(p, 4),
                         "Sig.": sig_star(p), "N": sub.shape[0]})

        chi2_df = pd.DataFrame(rows).sort_values("p-value")
        st.markdown("**Table 5 - Chi-square tests (binary predictors vs credit uptake)**")
        st.dataframe(chi2_df, use_container_width=True, hide_index=True)
        sig_preds = chi2_df[chi2_df["p-value"] < 0.05]["Predictor"].tolist()
        if sig_preds:
            finding(f"Only {', '.join(sig_preds)} reach statistical significance (p<0.05). "
                    f"Technology adoption and digital visibility - not gender or training - drive "
                    f"formal credit demand among Nigerian SMEs.")
        else:
            finding("No binary predictor reaches significance in the current filter selection. "
                    "Broaden the filters to restore the full-sample pattern.")
        st.markdown("""
Significance codes: `***` p<0.001 &nbsp; `**` p<0.01 &nbsp; `*` p<0.05
""")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OBJ 2 - CREDIT ACCESS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Obj 2: Credit Access":
    st.header("Objective 2: Determinants of Credit Access and Loan Approval")
    st.markdown("""
<div class="section-note">
Research question: What factors influence whether SMEs obtain credit after applying?<br>
Sample: Firms that applied for credit (N varies by filter).
</div>""", unsafe_allow_html=True)

    applied = dff[dff["credit_uptake"] == 1].copy()
    n_applied = len(applied)
    n_approved = int(applied["loan_approved"].sum())
    approval_rate = pct(applied["loan_approved"].dropna())
    st.info(f"Applicant sub-sample: **{n_applied:,} firms** | "
            f"Approved: **{n_approved}** | Approval rate: **{approval_rate}%**")
    finding(f"Of {n_applied} firms that applied for credit, {n_approved} were approved "
            f"({approval_rate}%). The high conditional approval rate indicates that self-exclusion "
            f"and discouragement - not rejection - are the primary barriers to credit access.")

    tab1, tab2 = st.tabs(["National analysis", "Zone breakdown"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Loan application outcomes")
            outcomes = {1.0: "Approved Full", 2.0: "Approved Partial",
                        3.0: "Rejected", 4.0: "Withdrawn"}
            oc_counts = applied["k20a1"].map(outcomes).value_counts().reset_index()
            oc_counts.columns = ["Outcome", "N"]
            oc_counts["% of applicants"] = (oc_counts["N"] / max(n_applied, 1) * 100).round(1)
            fig = px.pie(oc_counts, names="Outcome", values="N", hole=0.4,
                         color_discrete_sequence=["#27ae60", "#8cb369", "#e74c3c", "#95a5a6"],
                         template="plotly_white")
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Table 6 - Loan application outcomes**")
            st.dataframe(oc_counts, use_container_width=True, hide_index=True)

        with c2:
            st.subheader("Approval rate by firm characteristic")
            approval_feats = {"No audit": ("k21", 0), "Audited": ("k21", 1),
                              "No bank acct": ("k6", 0), "Bank account": ("k6", 1),
                              "No website": ("c22b", 0), "Has website": ("c22b", 1)}
            af_rows = []
            for lbl, (col, val) in approval_feats.items():
                sub = applied[applied[col] == val]["loan_approved"].dropna()
                if len(sub) > 0:
                    af_rows.append({"Category": lbl,
                                    "Approval rate (%)": round(sub.mean() * 100, 1),
                                    "N": len(sub)})
            af_df = pd.DataFrame(af_rows)
            fig = px.bar(af_df, x="Category", y="Approval rate (%)",
                         color="Approval rate (%)", color_continuous_scale="RdYlGn",
                         text="Approval rate (%)", template="plotly_white")
            fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(height=320, yaxis_range=[0, 100], showlegend=False,
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Table 7 - Approval rate by firm characteristic**")
            st.dataframe(af_df, use_container_width=True, hide_index=True)
            if not af_df.empty:
                web_with = af_df[af_df["Category"] == "Has website"]["Approval rate (%)"].values
                web_without = af_df[af_df["Category"] == "No website"]["Approval rate (%)"].values
                if len(web_with) and len(web_without):
                    finding(f"Firms with a business website are approved at {web_with[0]:.1f}% vs "
                            f"{web_without[0]:.1f}% for those without - a {web_with[0]-web_without[0]:.1f} pp "
                            f"advantage. Digital visibility acts as a credibility signal to lenders.")

        st.subheader("Collateral requirements")
        coll_data = {"Land / buildings": "k14a", "Equipment": "k14b",
                     "Accounts receivable": "k14c", "Personal assets": "k14d"}
        coll_rows = [{"Collateral type": lbl, "Required (%)": round(dff[col].mean() * 100, 1)}
                     for lbl, col in coll_data.items()]
        coll_df = pd.DataFrame(coll_rows)
        fig = px.bar(coll_df, x="Collateral type", y="Required (%)",
                     color="Required (%)", color_continuous_scale="Purples",
                     text="Required (%)", template="plotly_white")
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(height=300, yaxis_range=[0, 90], showlegend=False,
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Table 8 - Collateral types required (% of borrowers)**")
        st.dataframe(coll_df, use_container_width=True, hide_index=True)
        top_coll = coll_df.loc[coll_df["Required (%)"].idxmax()]
        finding(f"'{top_coll['Collateral type']}' is the most commonly required collateral "
                f"({top_coll['Required (%)']:.1f}% of borrowers). This asset-based lending model "
                f"systematically disadvantages small, female-owned, and young firms that lack formal land titles.")

    with tab2:
        st.subheader("Approval rate by zone")
        z_apr = []
        for zone in ZONES:
            sub = applied[applied["region_label"] == zone]
            if sub.shape[0] < 3:
                continue
            n_out = sub["loan_approved"].notna().sum()
            z_apr.append({
                "Zone": zone,
                "N applicants": sub.shape[0],
                "With outcome data": n_out,
                "Approved N": int(sub["loan_approved"].sum()),
                "Approval rate (%)": round(sub["loan_approved"].mean() * 100, 1) if n_out > 0 else np.nan,
                "Approval - audited (%)": round(sub[sub["k21"] == 1]["loan_approved"].mean() * 100, 1)
                    if (sub["k21"] == 1).any() else np.nan,
            })
        z_apr_df = pd.DataFrame(z_apr).sort_values("Approval rate (%)", ascending=False)
        fig = px.bar(z_apr_df, x="Zone", y="Approval rate (%)",
                     color="Zone", text="Approval rate (%)",
                     color_discrete_sequence=ZONE_COLORS, template="plotly_white")
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig.update_layout(height=350, showlegend=False, yaxis_range=[0, 115],
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Table 9 - Zone-level credit access summary**")
        st.dataframe(z_apr_df, use_container_width=True, hide_index=True)
        if not z_apr_df.empty:
            top_z = z_apr_df.iloc[0]
            bot_z = z_apr_df.iloc[-1]
            finding(f"{top_z['Zone']} leads on approval rate ({top_z['Approval rate (%)']:.1f}%), "
                    f"while {bot_z['Zone']} is lowest ({bot_z['Approval rate (%)']:.1f}%). "
                    f"Zones with denser financial infrastructure and higher audit penetration "
                    f"convert a larger share of applications into approvals.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OBJ 3 - FINANCING STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Obj 3: Financing Structure":
    st.header("Objective 3: Financing Structure of Nigerian SMEs")
    st.markdown("""
<div class="section-note">
Research question: How do Nigerian SMEs finance their operations and investments?<br>
Variables: Working capital sources (k3a–k3dgh) and fixed asset financing sources (k5a–k5f).
</div>""", unsafe_allow_html=True)

    wc_vars = {"k3a": "Internal funds", "k3bc": "Bank borrowing", "k3e": "Non-bank FI",
               "k3f": "Supplier credit", "k3dgh": "Informal/other"}
    fa_vars = {"k5a": "Internal funds", "k5bc": "Bank borrowing",
               "k5e": "Non-bank FI", "k5f": "Supplier credit"}

    tab1, tab2 = st.tabs(["National overview", "Zone comparison"])

    with tab1:
        c1, c2 = st.columns(2)
        wc_means = {v: round(dff[k].mean(), 2) for k, v in wc_vars.items()}
        fa_means = {v: round(dff[k].mean(), 2) for k, v in fa_vars.items()}

        with c1:
            st.subheader("Working capital sources")
            wc_df = pd.DataFrame(list(wc_means.items()), columns=["Source", "Mean (%)"])
            wc_df["Median (%)"] = [round(dff[k].median(), 2) for k in wc_vars]
            wc_df["Std Dev"] = [round(dff[k].std(), 2) for k in wc_vars]
            fig = px.bar(wc_df.sort_values("Mean (%)", ascending=True),
                         x="Mean (%)", y="Source", orientation="h",
                         color="Mean (%)", color_continuous_scale="Greens",
                         text="Mean (%)", template="plotly_white")
            fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(height=300, margin=dict(l=10, r=30, t=10, b=10), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Table 10 - Working capital financing sources**")
            st.dataframe(wc_df, use_container_width=True, hide_index=True)
            finding(f"Internal funds dominate working capital at {wc_means['Internal funds']:.1f}% "
                    f"(median {dff['k3a'].median():.0f}%). The median of zero for bank borrowing "
                    f"confirms that most firms use no formal bank finance at all - not just a little.")

        with c2:
            st.subheader("Fixed asset financing sources")
            fa_df = pd.DataFrame(list(fa_means.items()), columns=["Source", "Mean (%)"])
            fa_df["Median (%)"] = [round(dff[k].median(), 2) for k in fa_vars]
            fa_df["Std Dev"] = [round(dff[k].std(), 2) for k in fa_vars]
            fig = px.bar(fa_df.sort_values("Mean (%)", ascending=True),
                         x="Mean (%)", y="Source", orientation="h",
                         color="Mean (%)", color_continuous_scale="Blues",
                         text="Mean (%)", template="plotly_white")
            fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(height=280, margin=dict(l=10, r=30, t=10, b=10), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Table 11 - Fixed asset financing sources**")
            st.dataframe(fa_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Working capital financing by zone (heatmap)")
        hm_rows = []
        for zone in ZONES:
            sub = dff[dff["region_label"] == zone]
            row = {"Zone": zone}
            for k, v in wc_vars.items():
                row[v] = round(sub[k].mean(), 1)
            hm_rows.append(row)
        hm_df = pd.DataFrame(hm_rows).set_index("Zone")
        fig = px.imshow(hm_df, text_auto=".1f", color_continuous_scale="YlOrRd",
                        aspect="auto", template="plotly_white", labels={"color": "Mean %"})
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Table 12 - Zone-level working capital financing structure (%)**")
        st.dataframe(hm_df.reset_index(), use_container_width=True, hide_index=True)
        finding("Zones with bank borrowing means closest to zero - visible as the palest cells "
                "in the heatmap - are the most financially isolated. South West's relatively "
                "higher bank and supplier credit shares reflect its more developed commercial ecosystem.")

        bank_rows = []
        for zone in ZONES:
            sub = dff[dff["region_label"] == zone]
            bank_rows.append({
                "Zone": zone, "WC bank (%)": round(sub["k3bc"].mean(), 1),
                "FA bank (%)": round(sub["k5bc"].mean(), 1),
                "WC supplier (%)": round(sub["k3f"].mean(), 1),
            })
        bank_df = pd.DataFrame(bank_rows)
        fig = px.bar(bank_df.melt(id_vars="Zone"), x="Zone", y="value",
                     color="variable", barmode="group", template="plotly_white",
                     labels={"value": "Mean %", "variable": "Source"},
                     color_discrete_sequence=[COLORS["blue"], COLORS["teal"], COLORS["orange"]])
        fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Table 13 - Bank and supplier financing by zone**")
        st.dataframe(bank_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OBJ 4 - BARRIERS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Obj 4: Barriers":
    st.header("Objective 4: Barriers to Formal Credit Access")
    st.markdown("""
<div class="section-note">
Research question: What obstacles prevent SMEs from accessing formal finance?<br>
Obstacle severity scale: 0 = No obstacle → 4 = Very severe obstacle.
</div>""", unsafe_allow_html=True)

    k17_map = {1.0: "Sufficient capital", 2.0: "Complex procedures",
               3.0: "Unfavourable rates", 4.0: "Collateral too high",
               5.0: "Loan size/maturity", 6.0: "Did not expect approval", 7.0: "Other"}
    m1a_map = {1.0: "Access to Finance", 2.0: "Access to Land", 3.0: "Licensing/Permits",
               4.0: "Corruption", 5.0: "Courts", 6.0: "Crime/Theft", 7.0: "Customs/Trade",
               8.0: "Electricity", 9.0: "Workforce Skills", 10.0: "Labour Regulations",
               11.0: "Political Instability", 12.0: "Informal Sector",
               13.0: "Tax Administration", 14.0: "Tax Rates", 15.0: "Transport"}
    obs_cols = {"Access to Finance": "k30", "Electricity": "c30a",
                "Corruption": "j30f", "Political Instability": "j30e",
                "Informal Competition": "e30", "Transport": "d30a",
                "Tax Rates": "j30a", "Tax Administration": "j30b"}

    tab1, tab2 = st.tabs(["National level", "Zone breakdown"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Why SMEs did not apply for credit")
            k17_counts = dff["k17"].map(k17_map).value_counts().reset_index()
            k17_counts.columns = ["Reason", "N"]
            k17_counts["%"] = (k17_counts["N"] / max(dff["k17"].notna().sum(), 1) * 100).round(1)
            k17_counts = k17_counts.sort_values("N", ascending=True)
            fig = px.bar(k17_counts, x="N", y="Reason", orientation="h",
                         text="%", color="N", color_continuous_scale="Reds",
                         template="plotly_white")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(height=340, margin=dict(l=10, r=40, t=10, b=10), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Table 14 - Reasons for not applying for credit**")
            st.dataframe(k17_counts.sort_values("N", ascending=False),
                         use_container_width=True, hide_index=True)
            if not k17_counts.empty:
                top_reason = k17_counts.sort_values("N", ascending=False).iloc[0]
                finding(f"'{top_reason['Reason']}' is the most cited reason for not applying "
                        f"({top_reason['%']:.1f}%), but this largely reflects discouraged borrowers "
                        f"who self-exclude rather than genuine capital adequacy. The second-ranked "
                        f"reason reveals the true supply-side barrier: unfavourable interest rates.")

        with c2:
            st.subheader("Obstacle severity - national (0–4 scale)")
            obs_rows = [{"Obstacle": lbl,
                         "Mean score": round(dff[col].mean(), 3),
                         "Median": round(dff[col].median(), 1),
                         "% Major/Very Severe": round(
                             dff[dff[col].isin([3, 4])].shape[0] /
                             max(dff[col].notna().sum(), 1) * 100, 1)}
                        for lbl, col in obs_cols.items()]
            obs_df = pd.DataFrame(obs_rows).sort_values("Mean score", ascending=True)
            fig = px.bar(obs_df, x="Mean score", y="Obstacle", orientation="h",
                         color="Mean score", color_continuous_scale="Oranges",
                         text="Mean score", template="plotly_white")
            fig.add_vline(x=2, line_dash="dash", line_color="grey",
                          annotation_text="Moderate", annotation_position="top right")
            fig.update_traces(texttemplate="%{x:.2f}", textposition="outside")
            fig.update_coloraxes(showscale=False)
            fig.update_layout(height=340, xaxis_range=[0, 4.5],
                              margin=dict(l=10, r=40, t=10, b=10), yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Table 15 - Obstacle severity scores (national)**")
            st.dataframe(obs_df.sort_values("Mean score", ascending=False),
                         use_container_width=True, hide_index=True)
            if not obs_df.empty:
                top_obs = obs_df.sort_values("Mean score", ascending=False).iloc[0]
                finding(f"{top_obs['Obstacle']} is the most severe obstacle (mean {top_obs['Mean score']:.2f}/4), "
                        f"rated major or very severe by {top_obs['% Major/Very Severe']:.1f}% of firms. "
                        f"Energy costs suppress SME revenue and indirectly reduce credit demand "
                        f"by limiting growth aspirations.")

        st.subheader("Top 10 biggest overall obstacles")
        m1a_counts = dff["m1a"].map(m1a_map).value_counts().head(10).reset_index()
        m1a_counts.columns = ["Obstacle", "N"]
        m1a_counts["%"] = (m1a_counts["N"] / max(dff["m1a"].notna().sum(), 1) * 100).round(1)
        fig = px.bar(m1a_counts.sort_values("N", ascending=True),
                     x="N", y="Obstacle", orientation="h",
                     text="%", color="N", color_continuous_scale="Blues",
                     template="plotly_white")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig.update_layout(height=360, margin=dict(l=10, r=40, t=10, b=10), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Table 16 - Top 10 biggest obstacles cited by SMEs**")
        st.dataframe(m1a_counts, use_container_width=True, hide_index=True)
        if not m1a_counts.empty:
            top1 = m1a_counts.iloc[0]; top2 = m1a_counts.iloc[1] if len(m1a_counts) > 1 else None
            msg = (f"{top1['Obstacle']} ({top1['%']:.1f}%) and "
                   f"{top2['Obstacle']} ({top2['%']:.1f}%)" if top2 is not None else
                   f"{top1['Obstacle']} ({top1['%']:.1f}%)")
            finding(f"{msg} together account for the majority of 'biggest obstacle' citations, "
                    f"confirming the twin structural constraints choking Nigerian SME performance "
                    f"and credit demand.")

    with tab2:
        st.subheader("Obstacle severity heatmap - by zone")
        hz_rows = [{"Zone": zone, **{lbl: round(dff[dff["region_label"] == zone][col].mean(), 2)
                                     for lbl, col in obs_cols.items()}}
                   for zone in ZONES]
        hz_df = pd.DataFrame(hz_rows).set_index("Zone")
        fig = px.imshow(hz_df, text_auto=".2f", color_continuous_scale="Reds",
                        aspect="auto", zmin=0, zmax=4, template="plotly_white",
                        labels={"color": "Mean severity (0–4)"})
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Table 17 - Obstacle severity by geo-political zone**")
        st.dataframe(hz_df.reset_index(), use_container_width=True, hide_index=True)
        finding("South South and South East rate access to finance as their most severe constraint "
                "(means 2.88 and 2.75 respectively), despite being economically active zones. "
                "North East scores lowest across almost all obstacles, reflecting lower economic "
                "aspirations and engagement with formal markets rather than a benign environment.")

        st.subheader("Finance obstacle vs credit uptake by zone")
        scatter_rows = [{"Zone": z,
                         "Finance obstacle mean": round(dff[dff["region_label"] == z]["k30"].mean(), 2),
                         "Credit uptake (%)": pct(dff[dff["region_label"] == z]["credit_uptake"].dropna()),
                         "N": dff[dff["region_label"] == z].shape[0]}
                        for z in ZONES if dff[dff["region_label"] == z].shape[0] > 3]
        scat_df = pd.DataFrame(scatter_rows)
        fig = px.scatter(scat_df, x="Finance obstacle mean", y="Credit uptake (%)",
                         text="Zone", size="N", size_max=25,
                         color="Zone", color_discrete_sequence=ZONE_COLORS,
                         template="plotly_white",
                         labels={"Finance obstacle mean": "Mean finance obstacle severity (0–4)"})
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(height=370, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OBJ 5 & 6 - TRANSPARENCY & GENDER
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Obj 5 & 6: Transparency & Gender":
    st.header("Objectives 5 & 6: Financial Transparency and Gender")
    st.markdown("""
<div class="section-note">
Obj 5: Does financial transparency improve SMEs' access to credit? (k21, k6, c22b)<br>
Obj 6: Do female-owned or female-managed SMEs face different credit outcomes? (b4, b7a)
</div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Transparency effects", "Gender analysis", "Zone breakdown"])
    outcomes_list = {"Credit uptake": "credit_uptake", "Has credit": "has_credit",
                     "Loan approved": "loan_approved"}
    transparency_vars = {"External audit (k21)": "k21", "Bank account (k6)": "k6",
                         "Has website (c22b)": "c22b"}

    with tab1:
        st.subheader("Effect of transparency on credit outcomes")
        trans_rows = []
        for t_lbl, t_col in transparency_vars.items():
            for o_lbl, o_col in outcomes_list.items():
                sub = dff[[t_col, o_col]].dropna()
                if sub.shape[0] < 10:
                    continue
                r_without = round(sub[sub[t_col] == 0][o_col].mean() * 100, 1)
                r_with    = round(sub[sub[t_col] == 1][o_col].mean() * 100, 1)
                ct = pd.crosstab(sub[t_col], sub[o_col])
                chi2, p, _, _ = (chi2_contingency(ct) if ct.shape == (2, 2)
                                 else (np.nan, np.nan, None, None))
                trans_rows.append({
                    "Indicator": t_lbl.split("(")[0].strip(), "Outcome": o_lbl,
                    "Without (%)": r_without, "With (%)": r_with,
                    "Difference (pp)": round(r_with - r_without, 1),
                    "p-value": round(p, 4) if not np.isnan(p) else np.nan,
                    "Sig.": sig_star(p) if not np.isnan(p) else ""
                })

        trans_df = pd.DataFrame(trans_rows)
        sel_out = st.selectbox("Select credit outcome to visualise:",
                               list(outcomes_list.keys()), key="trans_out")
        sub_t = trans_df[trans_df["Outcome"] == sel_out]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Without", x=sub_t["Indicator"],
                             y=sub_t["Without (%)"], marker_color=COLORS["red"],
                             text=sub_t["Without (%)"].map(lambda v: f"{v:.1f}%"),
                             textposition="outside"))
        fig.add_trace(go.Bar(name="With", x=sub_t["Indicator"],
                             y=sub_t["With (%)"], marker_color=COLORS["green"],
                             text=sub_t["With (%)"].map(lambda v: f"{v:.1f}%"),
                             textposition="outside"))
        fig.update_layout(barmode="group", height=350, template="plotly_white",
                          yaxis_range=[0, 100], yaxis_title=f"% {sel_out}",
                          margin=dict(l=10, r=10, t=10, b=10), legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        if not sub_t.empty:
            best = sub_t.loc[sub_t["Difference (pp)"].abs().idxmax()]
            finding(f"For '{sel_out}', {best['Indicator']} shows the largest effect: "
                    f"firms with it score {best['With (%)']:.1f}% vs {best['Without (%)']:.1f}% "
                    f"without (+{best['Difference (pp)']:.1f} pp). "
                    f"{'Statistically significant.' if best['Sig.'] else 'Not statistically significant at this filter level.'}")
        st.markdown(f"**Table 18 - Transparency effects on '{sel_out}'**")
        st.dataframe(sub_t[["Indicator", "Without (%)", "With (%)", "Difference (pp)", "p-value", "Sig."]],
                     use_container_width=True, hide_index=True)
        st.markdown("**Full transparency effects table (all outcomes)**")
        st.dataframe(trans_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Gender gap in credit outcomes")
        gender_vars = {"Any female owner (b4)": "b4", "Female manager (b7a)": "b7a"}
        g_rows = []
        for g_lbl, g_col in gender_vars.items():
            for o_lbl, o_col in outcomes_list.items():
                sub = dff[[g_col, o_col]].dropna()
                if sub.shape[0] < 10:
                    continue
                r_m = round(sub[sub[g_col] == 0][o_col].mean() * 100, 1)
                r_f = round(sub[sub[g_col] == 1][o_col].mean() * 100, 1)
                ct = pd.crosstab(sub[g_col], sub[o_col])
                chi2, p, _, _ = (chi2_contingency(ct) if ct.shape == (2, 2)
                                 else (np.nan, np.nan, None, None))
                g_rows.append({
                    "Gender indicator": g_lbl.split("(")[0].strip(), "Outcome": o_lbl,
                    "Male/non-female (%)": r_m, "Female (%)": r_f,
                    "Gender gap (pp)": round(r_f - r_m, 1),
                    "p-value": round(p, 4) if not np.isnan(p) else np.nan,
                    "Sig.": sig_star(p) if not np.isnan(p) else ""
                })

        g_df = pd.DataFrame(g_rows)
        sel_g = st.selectbox("Gender indicator:", list(gender_vars.keys()), key="gen_sel")
        sub_g = g_df[g_df["Gender indicator"] == sel_g.split("(")[0].strip()]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Male / non-female", x=sub_g["Outcome"],
                             y=sub_g["Male/non-female (%)"], marker_color=COLORS["blue"],
                             text=sub_g["Male/non-female (%)"].map(lambda v: f"{v:.1f}%"),
                             textposition="outside"))
        fig.add_trace(go.Bar(name="Female", x=sub_g["Outcome"],
                             y=sub_g["Female (%)"], marker_color="#e91e8c",
                             text=sub_g["Female (%)"].map(lambda v: f"{v:.1f}%"),
                             textposition="outside"))
        fig.update_layout(barmode="group", height=350, template="plotly_white",
                          yaxis_range=[0, 100], margin=dict(l=10, r=10, t=10, b=10),
                          legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        finding("No gender gap reaches statistical significance at the national level. "
                "The small applicant sub-sample limits power to detect modest effects. "
                "Zone-level heterogeneity (Tab 3) reveals where female disadvantage is most acute.")
        st.markdown("**Table 19 - Gender gap in credit outcomes**")
        st.dataframe(g_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Gender credit uptake gap by zone")
        zg_rows = []
        for zone in ZONES:
            sub = dff[dff["region_label"] == zone]
            for g_lbl, g_col in gender_vars.items():
                r_0 = round(sub[sub[g_col] == 0]["credit_uptake"].mean() * 100, 1)
                r_1 = round(sub[sub[g_col] == 1]["credit_uptake"].mean() * 100, 1)
                zg_rows.append({"Zone": zone, "Indicator": g_lbl.split("(")[0].strip(),
                                "Non-female (%)": r_0, "Female (%)": r_1,
                                "Gap (pp)": round(r_1 - r_0, 1)})
        zg_df = pd.DataFrame(zg_rows)
        fig = px.bar(zg_df, x="Zone", y="Gap (pp)", color="Indicator", barmode="group",
                     color_discrete_sequence=["#e91e8c", "#9c27b0"],
                     template="plotly_white",
                     labels={"Gap (pp)": "Gender gap in credit uptake (pp)"},
                     title="Negative values = female firms apply less")
        fig.add_hline(y=0, line_color="grey", line_dash="dash")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        if not zg_df.empty:
            worst = zg_df.loc[zg_df["Gap (pp)"].idxmin()]
            best  = zg_df.loc[zg_df["Gap (pp)"].idxmax()]
            finding(f"The most pronounced female disadvantage is in {worst['Zone']} "
                    f"({worst['Indicator'].strip()}: {worst['Gap (pp)']:.1f} pp). "
                    f"South West uniquely shows a positive gap ({best['Gap (pp)']:.1f} pp for "
                    f"{best['Indicator'].strip()}), suggesting greater female economic empowerment "
                    f"in Lagos-centred commerce.")
        st.markdown("**Table 20 - Gender uptake gap (pp) by zone**")
        st.dataframe(zg_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OBJ 7 - FIRM PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Obj 7: Firm Performance":
    st.header("Objective 7: Firm Performance and Credit Access")
    st.markdown("""
<div class="section-note">
Research question: Are better-performing SMEs more likely to access formal finance?<br>
Performance variables: annual sales, employees, sales growth, capacity utilisation,
manager experience.
</div>""", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Distribution charts", "Zone analysis"])

    with tab1:
        perf_sel = st.selectbox(
            "Performance variable:",
            ["Annual sales (NGN)", "No. of employees", "Sales growth (%)",
             "Capacity utilisation (%)", "Manager experience (yrs)"]
        )
        perf_col = {"Annual sales (NGN)": "ln_sales", "No. of employees": "ln_employees",
                    "Sales growth (%)": "sales_growth", "Capacity utilisation (%)": "f1",
                    "Manager experience (yrs)": "b7"}[perf_sel]

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"{perf_sel} - by credit uptake")
            plot_df = dff[[perf_col, "credit_uptake"]].dropna().copy()
            if perf_col == "sales_growth":
                plot_df[perf_col] = plot_df[perf_col].clip(-100, 300)
            plot_df["Group"] = plot_df["credit_uptake"].map({0.0: "Did not apply", 1.0: "Applied"})
            fig = px.histogram(plot_df, x=perf_col, color="Group", barmode="overlay", nbins=25,
                               color_discrete_map={"Did not apply": COLORS["red"],
                                                   "Applied": COLORS["green"]},
                               opacity=0.65, template="plotly_white",
                               labels={perf_col: perf_sel})
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              legend_orientation="h")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader(f"{perf_sel} - by credit ownership")
            plot_df2 = dff[[perf_col, "has_credit"]].dropna().copy()
            if perf_col == "sales_growth":
                plot_df2[perf_col] = plot_df2[perf_col].clip(-100, 300)
            plot_df2["Group"] = plot_df2["has_credit"].map({0.0: "No credit", 1.0: "Has credit"})
            fig = px.box(plot_df2, x="Group", y=perf_col, color="Group",
                         color_discrete_map={"No credit": COLORS["red"],
                                             "Has credit": COLORS["green"]},
                         points=False, template="plotly_white",
                         labels={perf_col: perf_sel, "Group": ""})
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        finding("Performance variables do not significantly predict credit demand (uptake), "
                "but they do predict credit ownership: firms with higher number of employees and "
                "annual sales are significantly more likely to hold a bank loan. Lenders use "
                "observable size and revenue signals as primary underwriting criteria.")

        st.subheader("Mann-Whitney U tests - performance vs credit outcomes")
        all_perf = {"Annual sales (NGN)": "ln_sales", "No. of employees": "ln_employees",
                    "Sales growth (%)": "sales_growth", "Capacity utilisation (%)": "f1",
                    "Manager experience (yrs)": "b7"}
        mw_rows = []
        for lbl, col in all_perf.items():
            for o_lbl, o_col in {"Credit uptake": "credit_uptake", "Has credit": "has_credit"}.items():
                sub = dff[[col, o_col]].dropna()
                if sub.shape[0] < 10:
                    continue
                g0 = sub[sub[o_col] == 0][col]
                g1 = sub[sub[o_col] == 1][col]
                stat, p = mannwhitneyu(g1, g0, alternative="two-sided")
                mw_rows.append({
                    "Variable": lbl, "Outcome": o_lbl,
                    "Median - No": round(g0.median(), 3),
                    "Median - Yes": round(g1.median(), 3),
                    "Mann-Whitney U": round(stat, 0),
                    "p-value": round(p, 4), "Sig.": sig_star(p)
                })
        mw_df = pd.DataFrame(mw_rows)
        st.markdown("**Table 21 - Mann-Whitney U tests: performance vs credit outcomes**")
        st.dataframe(mw_df, use_container_width=True, hide_index=True)
        sig_has = mw_df[(mw_df["Outcome"] == "Has credit") & (mw_df["p-value"] < 0.05)]["Variable"].tolist()
        if sig_has:
            finding(f"Significant predictors of credit ownership: {', '.join(sig_has)}. "
                    f"None of the performance variables significantly predict credit demand, "
                    f"confirming the two-stage structure of Nigeria's credit market.")

    with tab2:
        st.subheader("Zone performance and credit profile")
        z_perf = []
        for zone in ZONES:
            sub = dff[dff["region_label"] == zone]
            if sub.shape[0] < 3:
                continue
            z_perf.append({
                "Zone": zone, "N": sub.shape[0],
                "Median annual sales": round(sub["ln_sales"].median(), 2),
                "Median employees": round(sub["l1"].median(), 0),
                "Median sales growth (%)": round(sub["sales_growth"].median(), 1),
                "Median capacity util (%)": round(sub["f1"].median(), 1),
                "Has credit (%)": pct(sub["has_credit"].dropna())
            })
        z_perf_df = pd.DataFrame(z_perf)
        fig = px.scatter(z_perf_df, x="Median annual sales", y="Has credit (%)",
                         text="Zone", size="N", size_max=28,
                         color="Zone", color_discrete_sequence=ZONE_COLORS,
                         template="plotly_white")
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        if not z_perf_df.empty:
            best_z = z_perf_df.loc[z_perf_df["Median annual sales"].idxmax()]
            finding(f"{best_z['Zone']} leads on median annual sales ({best_z['Median annual sales']:.2f}) "
                    f"and has a credit ownership rate of {best_z['Has credit (%)']:.1f}%, "
                    f"consistent with performance-credit co-movement at the zone level.")
        st.markdown("**Table 22 - Zone performance and credit profile**")
        st.dataframe(z_perf_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OBJ 8 & 9 - CREDITWORTHINESS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Obj 8 & 9: Creditworthiness":
    st.header("Objectives 8 & 9: Creditworthiness Indicators and ML Feature Selection")
    st.markdown("""
<div class="section-note">
Obj 8: Identify empirically derived creditworthiness indicators for Nigerian SMEs.<br>
Obj 9: Derive features for Machine Learning credit-scoring, loan approval, and loan amount models.
</div>""", unsafe_allow_html=True)

    candidates = {
        "Annual sales (NGN)": "ln_sales", "No. of employees": "ln_employees",
        "External audit": "k21", "Bank account": "k6", "Manager experience": "b7",
        "Firm age": "firm_age", "Capacity utilisation": "f1", "Sales growth": "sales_growth",
        "E-payments received %": "k33", "E-payments made %": "k38",
        "Female ownership": "b4", "WC internal funds %": "k3a",
        "WC bank funding %": "k3bc", "Formal training": "l10",
        "Product innovation": "h1", "Has website": "c22b",
    }
    cw_rows = []
    for lbl, col in candidates.items():
        sub = dff[["has_credit", col]].dropna()
        if sub.shape[0] < 20:
            continue
        r, p = pointbiserialr(sub["has_credit"], sub[col])
        cw_rows.append({
            "Feature": lbl, "Variable": col,
            "r (point-biserial)": round(r, 4),
            "p-value": round(p, 4), "Sig.": sig_star(p),
            "N": sub.shape[0],
            "ML priority": ("High" if abs(r) > 0.10 and p < 0.05 else
                            ("Medium" if abs(r) > 0.05 and p < 0.05 else "Low"))
        })
    cw_df = pd.DataFrame(cw_rows).sort_values("r (point-biserial)", ascending=False)

    tab1, tab2 = st.tabs(["Feature correlations", "Zone creditworthiness"])

    with tab1:
        st.subheader("Point-biserial correlations with credit ownership")
        sig_cw = cw_df[cw_df["p-value"] < 0.1].sort_values("r (point-biserial)")
        fig = go.Figure(go.Bar(
            x=sig_cw["r (point-biserial)"], y=sig_cw["Feature"],
            orientation="h",
            text=[f'{v:.3f} {s}' for v, s in zip(sig_cw["r (point-biserial)"], sig_cw["Sig."])],
            textposition="outside",
            marker_color=[COLORS["green"] if v >= 0 else COLORS["red"]
                          for v in sig_cw["r (point-biserial)"]]
        ))
        fig.add_vline(x=0, line_color="black", line_width=1)
        fig.update_layout(height=450, template="plotly_white",
                          xaxis_title="Point-biserial r",
                          margin=dict(l=10, r=60, t=10, b=10), yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        high_pri = cw_df[cw_df["ML priority"] == "High"]["Feature"].tolist()
        if high_pri:
            finding(f"High-priority ML features (|r| > 0.10, p < 0.05): {', '.join(high_pri)}. "
                    f"Note that WC bank funding % is likely endogenous - use with caution. "
                    f"The remaining features are exogenous and suitable for credit-scoring model inputs.")
        st.markdown("**Table 23 - Creditworthiness indicator correlations (all features)**")
        st.dataframe(cw_df, use_container_width=True, hide_index=True)

        st.subheader("ML model roadmap")
        roadmap = pd.DataFrame({
            "Model": ["Model 1", "Model 2", "Model 3"],
            "Objective": ["Creditworthiness prediction", "Loan approval prediction",
                          "Loan amount recommendation"],
            "Target variable": ["has_credit", "loan_approved", "k11 (loan value)"],
            "Top features": [
                "Annual sales, No. of employees, External audit, Bank account, Manager experience, Firm age",
                "Annual sales, External audit, Bank account, No. of employees, Collateral value",
                "Annual sales, No. of employees, External audit, Sales growth, Capacity utilisation"
            ],
            "Recommended algorithms": [
                "Logistic Regression, Random Forest, XGBoost",
                "Logistic Regression, Gradient Boosting",
                "Linear Regression, Ridge, Random Forest Regressor"
            ]
        })
        st.dataframe(roadmap, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Creditworthiness profile by zone")
        z_cw = []
        for zone in ZONES:
            sub = dff[dff["region_label"] == zone]
            if sub.shape[0] < 3:
                continue
            z_cw.append({
                "Zone": zone, "N": sub.shape[0],
                "Has credit (%)": pct(sub["has_credit"].dropna()),
                "Creditworthy proxy (%)": pct(sub["creditworthy"].dropna()),
                "Externally audited (%)": pct(sub["k21"].dropna()),
                "Has bank account (%)": pct(sub["k6"].dropna()),
                "Has website (%)": pct(sub["c22b"].dropna()),
                "Median annual sales": round(sub["ln_sales"].median(), 2),
            })
        z_cw_df = pd.DataFrame(z_cw).sort_values("Has credit (%)", ascending=False)
        metrics = ["Has credit (%)", "Creditworthy proxy (%)", "Externally audited (%)", "Has website (%)"]
        fig = px.bar(z_cw_df.melt(id_vars="Zone", value_vars=metrics),
                     x="Zone", y="value", color="variable", barmode="group",
                     color_discrete_sequence=[COLORS["green"], COLORS["teal"],
                                              COLORS["blue"], COLORS["purple"]],
                     template="plotly_white",
                     labels={"value": "% of firms", "variable": "Metric"})
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), legend_orientation="h")
        st.plotly_chart(fig, use_container_width=True)
        if not z_cw_df.empty:
            top_z = z_cw_df.iloc[0]; bot_z = z_cw_df.iloc[-1]
            finding(f"{top_z['Zone']} leads on credit ownership ({top_z['Has credit (%)']:.1f}%) "
                    f"and creditworthy proxy ({top_z['Creditworthy proxy (%)']:.1f}%). "
                    f"{bot_z['Zone']} trails across every metric, presenting the greatest "
                    f"challenge and opportunity for targeted financial inclusion interventions.")
        st.markdown("**Table 24 - Zone-level creditworthiness profile**")
        st.dataframe(z_cw_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ZONE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Zone Comparison":
    st.header("Geo-Political Zone Deep Comparison")
    st.markdown("""
<div class="section-note">
Side-by-side comparison of all key credit and firm characteristics across Nigeria's
six geo-political zones. Select two zones below for direct comparison.
</div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        zone_a = st.selectbox("Zone A", ZONES, index=0)
    with col_b:
        zone_b = st.selectbox("Zone B", ZONES, index=5)

    sub_a = dff[dff["region_label"] == zone_a]
    sub_b = dff[dff["region_label"] == zone_b]

    def zone_profile_dict(sub, name):
        return {
            "Zone": name, "N firms": sub.shape[0],
            "Credit uptake (%)": pct(sub["credit_uptake"].dropna()),
            "Has credit (%)": pct(sub["has_credit"].dropna()),
            "Approval rate (%)": pct(sub[sub["credit_uptake"] == 1]["loan_approved"].dropna()),
            "Externally audited (%)": pct(sub["k21"].dropna()),
            "Has website (%)": pct(sub["c22b"].dropna()),
            "Female-owned (%)": pct(sub["b4"].dropna()),
            "WC internal (%)": round(sub["k3a"].mean(), 1),
            "WC bank (%)": round(sub["k3bc"].mean(), 1),
            "Median annual sales": round(sub["ln_sales"].median(), 2),
            "Median employees": round(sub["l1"].median(), 0),
            "Electricity obstacle": round(sub["c30a"].mean(), 2),
            "Finance obstacle": round(sub["k30"].mean(), 2),
        }

    pa = zone_profile_dict(sub_a, zone_a)
    pb = zone_profile_dict(sub_b, zone_b)
    comp_df = pd.DataFrame([pa, pb]).set_index("Zone").T.reset_index()
    comp_df.columns = ["Metric", zone_a, zone_b]
    comp_df[zone_a] = pd.to_numeric(comp_df[zone_a], errors="coerce")
    comp_df[zone_b] = pd.to_numeric(comp_df[zone_b], errors="coerce")
    comp_df["Difference"] = (comp_df[zone_a] - comp_df[zone_b]).round(2)
    comp_df["Higher"] = comp_df.apply(
        lambda r: zone_a if r[zone_a] > r[zone_b] else (zone_b if r[zone_b] > r[zone_a] else "Equal"),
        axis=1
    )
    st.subheader(f"Direct comparison: {zone_a} vs {zone_b}")
    st.markdown("**Table 25 - Zone comparison scorecard**")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    plot_metrics = ["Credit uptake (%)", "Has credit (%)", "Externally audited (%)",
                    "Has website (%)", "Female-owned (%)", "WC bank (%)"]
    plot_vals = {zone_a: [pa[m] for m in plot_metrics],
                 zone_b: [pb[m] for m in plot_metrics]}
    fig = go.Figure()
    fig.add_trace(go.Bar(name=zone_a, x=plot_metrics, y=plot_vals[zone_a],
                         marker_color=ZONE_COLORS[ZONES.index(zone_a)],
                         text=[f"{v:.1f}%" for v in plot_vals[zone_a]], textposition="outside"))
    fig.add_trace(go.Bar(name=zone_b, x=plot_metrics, y=plot_vals[zone_b],
                         marker_color=ZONE_COLORS[ZONES.index(zone_b)],
                         text=[f"{v:.1f}%" for v in plot_vals[zone_b]], textposition="outside"))
    fig.update_layout(barmode="group", height=400, template="plotly_white",
                      margin=dict(l=10, r=10, t=10, b=40),
                      legend_orientation="h", xaxis_tickangle=-15, yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    # Dynamic finding for the comparison
    max_diff_row = comp_df.loc[comp_df["Difference"].abs().idxmax()]
    finding(f"The largest gap between {zone_a} and {zone_b} is on '{max_diff_row['Metric']}': "
            f"{zone_a} = {max_diff_row[zone_a]:.1f}, {zone_b} = {max_diff_row[zone_b]:.1f} "
            f"(difference: {max_diff_row['Difference']:.1f}).")

    st.subheader("Full zone summary table")
    all_zones = [zone_profile_dict(dff[dff["region_label"] == z], z)
                 for z in ZONES if dff[dff["region_label"] == z].shape[0] > 3]
    all_df = pd.DataFrame(all_zones)
    st.markdown("**Table 26 - All zones - key credit and firm indicators**")
    st.dataframe(all_df.sort_values("Credit uptake (%)", ascending=False),
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: NIGERIA ZONE MAP
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Nigeria Zone Map":
    st.header("Interactive Nigeria Geo-Political Zone Map")
    st.markdown("""
<div class="section-note">
Each bubble represents one of Nigeria's six geo-political zones, sized by number of firms
and coloured by the selected metric. Click the metric selector to switch views.
Hover over each zone for full details.
</div>""", unsafe_allow_html=True)

    metric_options = {
        "Credit uptake (%)": ("credit_uptake", "% firms that applied for credit"),
        "Credit access - Has loan (%)": ("has_credit", "% firms with an active bank loan"),
        "Creditworthiness proxy (%)": ("creditworthy", "% firms: has loan AND externally audited"),
        "Externally audited (%)": ("k21", "% firms with external audit"),
        "Has website (%)": ("c22b", "% firms with a business website"),
        "Loan approval rate (%)": ("loan_approved", "% of applicants approved"),
        "WC bank borrowing (%)": ("k3bc", "Mean % of WC from bank loans"),
        "Finance obstacle severity": ("k30", "Mean severity: 0=None, 4=Very Severe"),
        "Electricity obstacle severity": ("c30a", "Mean severity: 0=None, 4=Very Severe"),
    }
    selected_metric = st.selectbox("Colour zones by:", list(metric_options.keys()))
    col_key, col_desc = metric_options[selected_metric]

    # Use a fixed internal column key so the dict never gets a duplicate
    # when selected_metric happens to match one of the hard-coded column names
    # (e.g. "Credit uptake (%)" is both selected_metric and a fixed column).
    METRIC_COL_MAP = {
        "Credit uptake (%)":             "metric_credit_uptake",
        "Credit access - Has loan (%)":  "metric_has_credit",
        "Creditworthiness proxy (%)":    "metric_creditworthy",
        "Externally audited (%)":        "metric_audited",
        "Has website (%)":               "metric_website",
        "Loan approval rate (%)":        "metric_approval",
        "WC bank borrowing (%)":         "metric_wc_bank",
        "Finance obstacle severity":     "metric_finance_obs",
        "Electricity obstacle severity": "metric_elec_obs",
    }
    metric_col = METRIC_COL_MAP.get(selected_metric, "metric_val")

    map_rows = []
    for zone in ZONES:
        sub = dff[dff["region_label"] == zone]
        if sub.shape[0] == 0:
            continue
        lat, lon = ZONE_COORDS[zone]
        if col_key in ["k3bc", "k30", "c30a"]:
            val = round(sub[col_key].mean(), 2)
        else:
            val = pct(sub[col_key].dropna())

        map_rows.append({
            "Zone": zone, "Lat": lat, "Lon": lon,
            "N firms": sub.shape[0],
            metric_col:               val,
            "Credit uptake (%)":      pct(sub["credit_uptake"].dropna()),
            "Has credit (%)":         pct(sub["has_credit"].dropna()),
            "Creditworthy (%)":       pct(sub["creditworthy"].dropna()),
            "Approval rate (%)":      pct(sub[sub["credit_uptake"] == 1]["loan_approved"].dropna()),
            "Externally audited (%)": pct(sub["k21"].dropna()),
            "Has website (%)":        pct(sub["c22b"].dropna()),
            "WC bank (%)":            round(sub["k3bc"].mean(), 2),
            "Finance obstacle":       round(sub["k30"].mean(), 2),
        })

    map_df = pd.DataFrame(map_rows)

    if not map_df.empty:
        hover_data = {
            "N firms": True, "Credit uptake (%)": True, "Has credit (%)": True,
            "Creditworthy (%)": True, "Approval rate (%)": True,
            "Externally audited (%)": True, "Has website (%)": True,
            "WC bank (%)": True, "Finance obstacle": True,
            "Lat": False, "Lon": False, metric_col: True,
        }
        fig = px.scatter_mapbox(
            map_df, lat="Lat", lon="Lon",
            size="N firms", size_max=55,
            color=metric_col,
            color_continuous_scale="RdYlGn",
            hover_name="Zone",
            hover_data=hover_data,
            zoom=4.8, center={"lat": 9.0, "lon": 8.0},
            mapbox_style="carto-positron",
            template="plotly_white",
            title=f"Nigeria Zone Map - {selected_metric}."
        )
        fig.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0),
                          coloraxis_colorbar=dict(title=selected_metric[:25]))
        st.plotly_chart(fig, use_container_width=True)

        # Rename internal key to friendly label for the display table
        display_df = map_df[["Zone", "N firms", metric_col,
                              "Credit uptake (%)", "Has credit (%)", "Approval rate (%)"]].copy()
        display_df = display_df.rename(columns={metric_col: selected_metric})
        
        st.markdown("**Zone summary - selected metric**")
        display_df = display_df.loc[:, ~display_df.columns.duplicated()]
        st.dataframe(
            display_df.sort_values(selected_metric, ascending=False),
            use_container_width=True, hide_index=True
        )
        best  = map_df.loc[map_df[metric_col].idxmax()]
        worst = map_df.loc[map_df[metric_col].idxmin()]
        finding(f"On '{selected_metric}', {best['Zone']} scores highest ({best[metric_col]:.1f}) "
                f"and {worst['Zone']} lowest ({worst[metric_col]:.1f}) - "
                f"a {abs(best[metric_col] - worst[metric_col]):.1f}-point gap across Nigeria's zones.")
    else:
        st.warning("No zone data available for the current filter selection.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OBSTACLE INTELLIGENCE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Obstacle Intelligence":
    st.header("Credit Constraint Intelligence Dashboard")
    st.markdown("""
<div class="section-note">
Obstacle severity on a 0–4 scale (0 = No obstacle, 4 = Very Severe). All charts respond
dynamically to the zone, sector, and size filters in the sidebar.
</div>""", unsafe_allow_html=True)

    obstacle_map = {
        "Electricity":         "c30a",
        "Access to Finance":   "k30",
        "Political Instability": "j30e",
        "Corruption":          "j30f",
        "Tax Administration":  "j30b",
        "Transport":           "d30a",
        "Tax Rates":           "j30a",
        "Informal Competition":"e30",
    }

    # ── Ranked obstacle bar ────────────────────────────────────────────────────
    st.subheader("Obstacle Ranking - Mean Severity")
    rank_rows = []
    for label, col in obstacle_map.items():
        vals = dff[col].dropna()
        if len(vals) == 0:
            continue
        rank_rows.append({
            "Obstacle": label,
            "Mean severity": round(vals.mean(), 3),
            "Median": round(vals.median(), 1),
            "% Major / Very Severe": round((vals >= 3).mean() * 100, 1),
            "N": len(vals),
        })
    rank_df = pd.DataFrame(rank_rows).sort_values("Mean severity", ascending=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        for i, row in rank_df.iterrows():
            color = ("#c0392b" if row["Mean severity"] >= 2.5 else
                     "#e67e22" if row["Mean severity"] >= 2.0 else
                     "#2980b9" if row["Mean severity"] >= 1.5 else "#27ae60")
            fig.add_trace(go.Bar(
                x=[row["Mean severity"]], y=[row["Obstacle"]],
                orientation="h", marker_color=color, showlegend=False,
                text=[f"{row['Mean severity']:.2f}"], textposition="outside",
                name=row["Obstacle"]
            ))
        fig.add_vline(x=2, line_dash="dash", line_color="grey",
                      annotation_text="Moderate threshold", annotation_position="top right")
        fig.update_layout(height=380, template="plotly_white", xaxis_range=[0, 4.5],
                          xaxis_title="Mean severity (0–4)", yaxis_title="",
                          margin=dict(l=10, r=50, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Severity ranking**")
        st.dataframe(rank_df[["Obstacle", "Mean severity", "% Major / Very Severe"]],
                     use_container_width=True, hide_index=True)

    if not rank_df.empty:
        top_obs = rank_df.iloc[0]
        finding(f"**{top_obs['Obstacle']}** is the most severe obstacle "
                f"(mean {top_obs['Mean severity']:.2f}/4; rated Major or Very Severe by "
                f"{top_obs['% Major / Very Severe']:.1f}% of firms in this filter selection).")

    st.divider()

    # ── Obstacle severity by zone ──────────────────────────────────────────────
    st.subheader("Obstacle Severity by Zone - Heatmap")
    hz_rows = [{"Zone": zone, **{lbl: round(dff[dff["region_label"] == zone][col].mean(), 2)
                                 for lbl, col in obstacle_map.items()}}
               for zone in ZONES]
    hz_df = pd.DataFrame(hz_rows).set_index("Zone")
    fig = px.imshow(hz_df, text_auto=".2f", color_continuous_scale="RdYlGn_r",
                    aspect="auto", zmin=0, zmax=4, template="plotly_white",
                    labels={"color": "Severity (0–4)"})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Table - Obstacle severity by zone**")
    st.dataframe(hz_df.reset_index(), use_container_width=True, hide_index=True)
    finding("Darker red cells (higher severity) reveal where structural constraints are most acute. "
            "South East and South South consistently rate finance access and electricity as severe. "
            "North East scores lowest - reflecting lower economic engagement rather than a benign environment.")

    st.divider()

    # ── Severity distribution per obstacle ────────────────────────────────────
    st.subheader("Severity Distribution by Obstacle")
    sel_obs = st.selectbox("Select obstacle to explore:", list(obstacle_map.keys()),
                           key="obs_detail")
    obs_col = obstacle_map[sel_obs]
    dist_data = dff[obs_col].dropna().value_counts().sort_index().reset_index()
    dist_data.columns = ["Severity", "N firms"]
    dist_data["Severity label"] = dist_data["Severity"].map(
        {0: "No obstacle", 1: "Minor", 2: "Moderate", 3: "Major", 4: "Very Severe"})
    dist_data["% of firms"] = (dist_data["N firms"] / dist_data["N firms"].sum() * 100).round(1)
    fig = px.bar(dist_data, x="Severity label", y="N firms",
                 color="Severity",
                 color_continuous_scale=[[0, "#27ae60"], [0.5, "#f39c12"], [1, "#c0392b"]],
                 text="% of firms", template="plotly_white",
                 labels={"Severity label": "Severity", "N firms": "Firms"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Mean severity", f"{dff[obs_col].mean():.2f} / 4")
    with c2:
        pct_severe = round((dff[obs_col] >= 3).mean() * 100, 1)
        st.metric("% Major or Very Severe", f"{pct_severe}%")
    with c3:
        st.metric("Median severity", f"{dff[obs_col].median():.0f} / 4")
    finding(f"For {sel_obs}: mean severity {dff[obs_col].mean():.2f}/4; "
            f"{pct_severe:.0f}% of firms rate it Major or Very Severe. "
            f"Use the sidebar zone/sector/size filters to examine whether burden is "
            f"concentrated in particular sub-groups.")

    # ── Biggest obstacle distribution ─────────────────────────────────────────
    st.divider()
    st.subheader("Biggest Single Obstacle - Distribution")
    m1a_map = {1.0: "Access to Finance", 2.0: "Access to Land", 3.0: "Licensing/Permits",
               4.0: "Corruption", 5.0: "Courts", 6.0: "Crime/Theft", 7.0: "Customs/Trade",
               8.0: "Electricity", 9.0: "Workforce Skills", 10.0: "Labour Regulations",
               11.0: "Political Instability", 12.0: "Informal Sector",
               13.0: "Tax Administration", 14.0: "Tax Rates", 15.0: "Transport"}
    m1a_counts = dff["m1a"].map(m1a_map).value_counts().head(10).reset_index()
    m1a_counts.columns = ["Obstacle", "N"]
    m1a_counts["%"] = (m1a_counts["N"] / max(dff["m1a"].notna().sum(), 1) * 100).round(1)
    fig = px.bar(m1a_counts.sort_values("N", ascending=True), x="N", y="Obstacle",
                 orientation="h", text="%", color="N",
                 color_continuous_scale="Blues", template="plotly_white")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=360, margin=dict(l=10, r=40, t=10, b=10), yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(m1a_counts, use_container_width=True, hide_index=True)
    if not m1a_counts.empty:
        top1 = m1a_counts.iloc[0]
        finding(f"'{top1['Obstacle']}' is cited as the single biggest obstacle by "
                f"{top1['%']:.1f}% of firms in the current selection. "
                f"Policy interventions targeting this constraint would directly address "
                f"the most pervasive growth barrier for Nigerian SMEs.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EXECUTIVE REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📄 Executive Report":
    st.header("Executive Report Generator")
    st.markdown("""
<div class="section-note">
Generate a professional one-page executive summary of the SME Credit Research findings,
dynamically computed from the current filter selection. Export as a plain-text report
or download as a formatted text file.
</div>""", unsafe_allow_html=True)

    today = datetime.date.today().strftime("%d %B %Y")
    zone_str = ", ".join(zone_filter) if zone_filter else "All zones"
    sector_str = ", ".join(sector_filter) if sector_filter else "All sectors"
    size_str = ", ".join(size_filter) if size_filter else "All sizes"

    # ── Compute all stats for the report ──────────────────────────────────────
    n_total = len(dff)
    uptake_rate = pct(dff["credit_uptake"].dropna())
    has_credit_rate = pct(dff["has_credit"].dropna())
    approval_rate = pct(dff["loan_approved"].dropna())
    wc_internal = round(dff["k3a"].mean(), 1)
    wc_bank = round(dff["k3bc"].mean(), 1)
    pct_audited = pct(dff["k21"].dropna())
    pct_website = pct(dff["c22b"].dropna())
    pct_female_owned = pct(dff["b4"].dropna())

    # Best/worst zones
    zone_stats = {z: pct(dff[dff["region_label"] == z]["credit_uptake"].dropna()) for z in ZONES}
    zone_stats = {k: v for k, v in zone_stats.items() if dff[dff["region_label"] == k].shape[0] > 0}
    best_zone = max(zone_stats, key=zone_stats.get) if zone_stats else "N/A"
    worst_zone = min(zone_stats, key=zone_stats.get) if zone_stats else "N/A"

    # Obstacle rankings
    obs_means = {k: round(dff[v].mean(), 2) for k, v in
                 {"Electricity": "c30a", "Access to Finance": "k30",
                  "Political Instability": "j30e", "Corruption": "j30f",
                  "Tax Administration": "j30b"}.items()}
    top_obstacle = max(obs_means, key=obs_means.get)

    # Creditworthiness top features
    cw_results = []
    for lbl, col in [("No. of employees", "ln_employees"), ("Annual sales (NGN)", "ln_sales"),
                     ("Has website", "c22b"), ("External audit", "k21"),
                     ("E-payments made %", "k38")]:
        sub = dff[["has_credit", col]].dropna()
        if sub.shape[0] < 20:
            continue
        r, p = pointbiserialr(sub["has_credit"], sub[col])
        if p < 0.05:
            cw_results.append(f"{lbl} (r={r:.3f})")

    # ── Build report text ──────────────────────────────────────────────────────
    report_text = f"""SME CREDIT RESEARCH IN NIGERIA
World Bank Enterprise Survey 2025
═══════════════════════════════════════════════════════════════

EXECUTIVE REPORT - GENERATED: {today}

FILTER APPLIED
  Zones   : {zone_str}
  Sectors : {sector_str}
  Sizes   : {size_str}
  N firms : {n_total:,}

═══════════════════════════════════════════════════════════════
KEY FINDINGS
═══════════════════════════════════════════════════════════════

1. CREDIT DEMAND IS STRUCTURALLY SUPPRESSED
   Only {uptake_rate:.1f}% of firms applied for formal credit. This is not a
   rejection problem: {approval_rate:.0f}% of applicants were approved. The gap
   reflects self-exclusion driven by high rates, complex procedures,
   and embedded expectations of failure.

2. SELF-FINANCING IS NEAR-UNIVERSAL
   Internal funds cover {wc_internal:.1f}% of working capital; bank
   borrowing accounts for just {wc_bank:.1f}%. The median firm uses zero
   formal bank finance for working capital or fixed assets.

3. CREDIT STOCK IS LIMITED
   Only {has_credit_rate:.1f}% of firms hold an active bank loan. Formal
   credit remains concentrated in a small segment of the SME population.

4. DIGITAL PRESENCE AND AUDIT ARE THE KEY CREDIT ENABLERS
   Firms with a business website and external audit are significantly
   more likely to apply for and obtain credit. These characteristics
   reduce information asymmetry between SMEs and lenders.
   - Externally audited: {pct_audited:.1f}% of firms
   - Has website:        {pct_website:.1f}% of firms

5. GEOGRAPHIC DISPARITY IS STARK
   {best_zone} leads credit uptake; {worst_zone} trails significantly.
   Zone-specific interventions are essential - a national average
   policy will miss the most excluded populations.

6. ELECTRICITY IS THE PRIMARY STRUCTURAL BARRIER
   {top_obstacle} is the most severely rated business obstacle,
   indirectly suppressing credit demand by limiting SME revenue
   growth and capital investment aspirations.

7. GENDER GAPS ARE ZONE-SPECIFIC, NOT NATIONAL
   No significant gender gap exists at the national level. North West
   and South East show pronounced female disadvantage in credit uptake
   (gaps up to -13.3 pp), requiring targeted zone-level interventions.

8. CREDITWORTHINESS INDICATORS FOR ML MODELS
   Top predictors of credit ownership (point-biserial, p<0.05):
   {chr(10).join('   - ' + f for f in cw_results) if cw_results else '   - Insufficient data in current filter'}

═══════════════════════════════════════════════════════════════
RECOMMENDATIONS
═══════════════════════════════════════════════════════════════

FOR FINANCIAL INSTITUTIONS
  • Accept digital transaction history (e-payment records) as
    alternative credit evidence for SMEs lacking physical collateral.
  • Reduce dependence on land/buildings as collateral - currently
    required by 62.6% of lenders, which excludes asset-poor firms.
  • Streamline applications: 'complex procedures' (12.5%) and
    'unfavourable rates' (30.7%) are the leading stated barriers.

FOR POLICYMAKERS
  • Prioritise electricity infrastructure reform - the top obstacle
    for {obs_means.get('Electricity', 0):.2f}/4 mean severity nationally.
  • Implement zone-specific credit inclusion programmes targeting
    {worst_zone} and other low-uptake northern zones.
  • Promote SME formalisation (audit, registration, digital presence)
    as direct credit-enabling instruments, not merely compliance items.

FOR FUTURE RESEARCH
  • Develop and validate ML models using WBES-derived features
    (Model 1: creditworthiness; Model 2: approval; Model 3: loan amount).
  • Conduct qualitative follow-up with discouraged borrowers to
    distinguish genuine capital adequacy from strategic self-exclusion.

═══════════════════════════════════════════════════════════════
SOURCE
Dataset  : World Bank Enterprise Survey Nigeria 2025
N firms  : {n_total:,} (filtered selection)
Analysis : SME Credit Research - Insight Analytics Consult
Generated: {today}
═══════════════════════════════════════════════════════════════
"""

    # ── Display report in app ──────────────────────────────────────────────────
    st.subheader(f"Executive Report - Generated: {today}")
    st.code(report_text, language=None)

    # ── Download as .txt ──────────────────────────────────────────────────────
    st.download_button(
        label="📥 Download report as .txt",
        data=report_text.encode("utf-8"),
        file_name=f"SME_Credit_Research_Report_{today.replace(' ', '_')}.txt",
        mime="text/plain"
    )

    # ── Download as .csv summary table ────────────────────────────────────────
    summary_rows = [
        ["Metric", "Value", "Filter applied"],
        ["Report date", today, zone_str],
        ["N firms (filtered)", n_total, sector_str],
        ["Credit uptake (%)", uptake_rate, size_str],
        ["Has credit (%)", has_credit_rate, ""],
        ["Loan approval rate (%)", approval_rate, ""],
        ["WC internal funding (%)", wc_internal, ""],
        ["WC bank funding (%)", wc_bank, ""],
        ["Externally audited (%)", pct_audited, ""],
        ["Has website (%)", pct_website, ""],
        ["Female-owned (%)", pct_female_owned, ""],
        ["Best zone - credit uptake", best_zone, ""],
        ["Worst zone - credit uptake", worst_zone, ""],
        ["Top obstacle", top_obstacle, f"Mean {obs_means.get(top_obstacle, 0):.2f}/4"],
    ]
    csv_buf = io.StringIO()
    import csv
    writer = csv.writer(csv_buf)
    writer.writerows(summary_rows)
    st.download_button(
        label="📥 Download summary as .csv",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name=f"SME_Credit_Summary_{today.replace(' ', '_')}.csv",
        mime="text/csv"
    )

    st.info(
        "**Note on PDF/DOCX export:** Streamlit's deployment environment does not support "
        "in-browser PDF or Word generation without additional server-side libraries. "
        "To produce a formatted DOCX or PDF: (1) copy the text above into Word and save as .docx, "
        "or (2) install the `python-docx` and `reportlab` packages and run the companion "
        "`build_report.js` script from the project repository for a fully formatted output."
    )
