# SMEs Credit Research - Streamlit Web App
## World Bank Enterprise Survey Nigeria 2025

### Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

### How to connect your real data
Replace the `generate_data()` function in `app.py` with:
```python
@st.cache_data
def load_data():
    return pd.read_csv("wbes_nigeria_cleaned.csv")
df = load_data()
```
The cleaned CSV is produced by Section 16 of the notebook
(`SMEs_Credit_Research_Complete.ipynb`).

### Pages
| Page | Coverage |
|------|----------|
| Overview | KPI cards, zone uptake bar, WC pie, objectives summary |
| Obj 1: Credit Uptake | Sector/size/zone charts, formalisation enablers, chi-square table |
| Obj 2: Credit Access | Outcome pie, approval drivers, collateral, zone approval rates |
| Obj 3: Financing Structure | WC/FA source bars, zone heatmap, bank vs supplier by zone |
| Obj 4: Barriers | k17 reasons chart, obstacle severity bar + heatmap, scatter |
| Obj 5 & 6: Transparency & Gender | Transparency uplift, gender gap, zone gender breakdown |
| Obj 7: Firm Performance | Histograms/boxplots, Mann-Whitney table, zone scatter |
| Obj 8 & 9: Creditworthiness | Correlation bar, ML roadmap table, zone creditworthiness profile |
| Zone Comparison | Side-by-side zone scorecard + grouped bar for any two zones |

### Sidebar filters
All charts and tables respond live to:
- **Geo-political zone** (multi-select)
- **Sector** (multi-select)
- **Firm size** (multi-select)
