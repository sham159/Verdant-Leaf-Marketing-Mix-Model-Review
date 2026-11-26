# Step 3 - a simpler but better model than v0
import pandas as pd
import numpy as np
import statsmodels.api as sm

d = pd.read_csv("data/mmm_weekly_clean.csv", parse_dates=["week_start"])
d["t"] = np.arange(len(d))          # week number = simple growth trend

channels = ["meta_spend", "google_search_spend", "youtube_spend",
            "influencer_spend", "tv_spend", "amazon_ads_spend"]
promo = ["marketplace_sale_flag", "competitor_promo_flag", "new_launch_flag"]

def adstock(x, decay):
    # carryover: this week's effect = this week's spend + decay * last week's effect
    out, prev = [], 0
    for v in x:
        prev = v + decay * prev
        out.append(prev)
    return np.array(out)

def make_X(df, scale, decay, saturate, extra):
    X = pd.DataFrame(index=df.index)
    for c in channels:
        a = adstock(df[c].values, decay) / scale[c]
        X[c] = np.log1p(a) if saturate else a      # log = each extra rupee earns less
    return pd.concat([X, df[extra]], axis=1)

def test_mape(decay, saturate, extra, cut, end):
    # fit on weeks before `cut`, predict the year after it
    train = d[d["week_start"] < cut]
    test = d[(d["week_start"] >= cut) & (d["week_start"] < end)]
    scale = {c: train[c].mean() for c in channels}
    m = sm.OLS(train["revenue_inr"], sm.add_constant(make_X(train, scale, decay, saturate, extra))).fit()
    p = m.predict(sm.add_constant(make_X(test, scale, decay, saturate, extra), has_constant="add"))
    return ((test["revenue_inr"] - p).abs() / test["revenue_inr"]).mean() * 100

# ---- compare a few versions on weeks the model has not seen ----
v0_controls = ["festive_flag", "avg_discount_pct"]
ctrl = ["festive_flag"] + promo + ["t"]      # discount dropped: 0.86 correlated with marketplace_sale_flag
specs = {
    "1. v0 spec (cleaned data)":            (0,  False, v0_controls),
    "2. + promo flags":                     (0,  False, v0_controls + promo),
    "3. + promo flags + trend":             (0,  False, v0_controls + promo + ["t"]),
    "4. 3 + carryover (decay 0.3)":         (.3, False, v0_controls + promo + ["t"]),
    "5. 4 + diminishing returns (log)":     (.3, True,  v0_controls + promo + ["t"]),
    "6. final: 5 without discount":         (.3, True,  ctrl),
    "7. final + brand search (sensitivity)": (.3, True, ctrl + ["brand_search_index"]),
}
print("average % error predicting 2024 (trained on 2023) and 2025 (trained on 2023-24)")
for name, (dec, sat, ex) in specs.items():
    a = test_mape(dec, sat, ex, "2024-01-01", "2025-01-01")
    b = test_mape(dec, sat, ex, "2025-01-01", "2026-01-01")
    print(f"{name:<40} 2024: {a:4.1f}  2025: {b:4.1f}")

print("\nhow sensitive is the final model to the carryover decay?")
for dec in [0, .2, .3, .4, .5]:
    a = test_mape(dec, True, ctrl, "2024-01-01", "2025-01-01")
    b = test_mape(dec, True, ctrl, "2025-01-01", "2026-01-01")
    print(f"  decay {dec}: 2024 {a:4.1f}  2025 {b:4.1f}")

# ---- final model on all 156 weeks ----
def fit_final(extra):
    scale = {c: d[c].mean() for c in channels}
    m = sm.OLS(d["revenue_inr"], sm.add_constant(make_X(d, scale, .3, True, extra))).fit()
    return m, scale

m, scale = fit_final(ctrl)
print(m.summary().tables[1])
print("R2:", round(m.rsquared, 3), " Durbin-Watson:", round(sm.stats.durbin_watson(m.resid), 2))

# ---- marginal return: extra revenue per extra Rs 1 at 2025 average weekly spend ----
# derivative of b*log(1 + adstock/scale); in steady state adstock = spend / (1 - decay)
def marginal_roi(model, scale, decay=.3):
    x25 = d[d["week_start"] >= "2025-01-01"][channels].mean()
    ci = model.conf_int()
    rows = {}
    for c in channels:
        den = scale[c] * (1 - decay) + x25[c]
        rows[c] = [model.params[c] / den, ci.loc[c, 0] / den, ci.loc[c, 1] / den]
    return pd.DataFrame(rows, index=["marginal_roi", "low_95", "high_95"]).T.round(2)

print("\nmarginal ROI, final model:"); print(marginal_roi(m, scale))
m7, _ = fit_final(ctrl + ["brand_search_index"])
print("\nmarginal ROI, with brand search added (sensitivity):"); print(marginal_roi(m7, scale))

# is brand search driven by ads? (decides whether it is a fair control)
Xb = make_X(d, scale, .3, True, ["t"])
mb = sm.OLS(d["brand_search_index"], sm.add_constant(Xb)).fit()
print("\nbrand search regressed on ads (t-values):"); print(mb.tvalues[channels].round(1))
