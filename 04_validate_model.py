# Step 4 - is the model credible, or just a good fit?
import pandas as pd
import numpy as np
import statsmodels.api as sm

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

d = pd.read_csv("data/mmm_weekly_clean.csv", parse_dates=["week_start"])
d["t"] = np.arange(len(d))
channels = ["meta_spend", "google_search_spend", "youtube_spend",
            "influencer_spend", "tv_spend", "amazon_ads_spend"]
promo = ["marketplace_sale_flag", "competitor_promo_flag", "new_launch_flag"]
ctrl = ["festive_flag"] + promo + ["t"]            # final model controls (Step 3)

def adstock(x, decay):
    out, prev = [], 0
    for v in x:
        prev = v + decay * prev
        out.append(prev)
    return np.array(out)

def make_X(df, scale, decay=.3, saturate=True, extra=ctrl):
    X = pd.DataFrame(index=df.index)
    for c in channels:
        a = adstock(df[c].values, decay) / scale[c]
        X[c] = np.log1p(a) if saturate else a
    return pd.concat([X, df[extra]], axis=1)

def fit(mask, decay=.3, saturate=True, extra=ctrl, df=None):
    df = d if df is None else df
    scale = {c: df.loc[mask, c].mean() for c in channels}
    X = make_X(df, scale, decay, saturate, extra)
    return sm.OLS(df.loc[mask, "revenue_inr"], sm.add_constant(X[mask])).fit(), scale, X

def marginal_roi(m, scale, decay=.3, saturate=True):
    # extra revenue per extra Rs 1 of weekly spend, at 2025 average spend
    x25 = d[d["week_start"] >= "2025-01-01"][channels].mean()
    out = {}
    for c in channels:
        den = scale[c] * (1 - decay) + x25[c] if saturate else scale[c] * (1 - decay)
        out[c] = m.params[c] / den
    return pd.Series(out)

# ===== 1. predicting future weeks: 8 rolling 13-week tests =====
print("1. ROLLING TIME TESTS (train on everything before, predict next 13 weeks)")
v0_extra = ["festive_flag", "avg_discount_pct"]
naive = d["revenue_inr"].shift(52)                  # "same week last year"
rows = []
for cut in pd.date_range("2024-01-01", "2025-10-01", freq="3MS"):
    train = d["week_start"] < cut
    test = (d["week_start"] >= cut) & (d["week_start"] < cut + pd.Timedelta(weeks=13))
    mf, sc, Xf = fit(train)
    pf = mf.predict(sm.add_constant(Xf[test], has_constant="add"))
    m0, sc0, X0 = fit(train, decay=0, saturate=False, extra=v0_extra)
    p0 = m0.predict(sm.add_constant(X0[test], has_constant="add"))
    y = d.loc[test, "revenue_inr"]
    mape = lambda p: ((y - p).abs() / y).mean() * 100
    rows.append([cut.date(), test.sum(), mape(pf), mape(p0), mape(naive[test])])
res = pd.DataFrame(rows, columns=["test_start", "weeks", "final", "v0_clean", "same_week_last_year"]).round(1)
print(res.to_string(index=False)); print(res[["final", "v0_clean", "same_week_last_year"]].mean().round(1).to_dict())

# ===== 2. residuals (errors of the model on all 156 weeks) =====
print("\n2. RESIDUALS")
mf, scale, X = fit(d["t"] >= 0)
r = mf.resid
print("R2", round(mf.rsquared, 3), "| Durbin-Watson", round(sm.stats.durbin_watson(r), 2),
      "| lag-1 autocorrelation", round(r.autocorr(1), 2))
print("avg error % by year:", (r.abs() / d["revenue_inr"]).groupby(d["week_start"].dt.year).mean().mul(100).round(1).to_dict())
pct = r.abs() / d["revenue_inr"] * 100
for f in ["festive_flag", "marketplace_sale_flag"]:
    print(f"avg error % in {f}=1 weeks:", round(pct[d[f] == 1].mean(), 1), "| other weeks:", round(pct[d[f] == 0].mean(), 1))
print("5 biggest misses:"); print(pd.DataFrame({"week": d["week_start"].dt.date, "miss_lakh": (r / 1e5).round(0)}).loc[r.abs().nlargest(5).index].to_string(index=False))
print("residual vs brand_search (not in model):", round(r.corr(d["brand_search_index"]), 2),
      "| vs organic traffic:", round(r.corr(d["organic_traffic_index"]), 2))

# lead test: does NEXT week's spend help explain THIS week's revenue? (should not, if ads cause sales)
lead = pd.DataFrame({c + "_next": np.log1p(d[c].shift(-1) / scale[c]) for c in channels})
Xl = pd.concat([X, lead], axis=1).dropna()
ml = sm.OLS(d.loc[Xl.index, "revenue_inr"], sm.add_constant(Xl)).fit()
print("\nlead test (t-values of NEXT week's spend):", ml.tvalues[lead.columns].round(1).to_dict())

# ===== 3. stability: do channel returns change when we change the data window? =====
print("\n3. STABILITY - marginal ROI at 2025 spend, fitted on different windows")
wins = {"2023-24 only": d["week_start"] < "2025-01-01",
        "2024-25 only": d["week_start"] >= "2024-01-01",
        "all weeks": d["t"] >= 0}
print(pd.DataFrame({k: marginal_roi(*fit(v)[:2]) for k, v in wins.items()}).round(2))

print("\nsensitivity to modelling choices (all weeks)")
sens = {"decay 0.2": marginal_roi(*fit(wins["all weeks"], decay=.2)[:2], decay=.2),
        "decay 0.3 (final)": marginal_roi(mf, scale),
        "decay 0.4": marginal_roi(*fit(wins["all weeks"], decay=.4)[:2], decay=.4),
        "decay 0.5": marginal_roi(*fit(wins["all weeks"], decay=.5)[:2], decay=.5),
        "no saturation (linear)": marginal_roi(*fit(wins["all weeks"], saturate=False)[:2], saturate=False),
        "+ brand search": marginal_roi(*fit(wins["all weeks"], extra=ctrl + ["brand_search_index"])[:2])}
d2 = d.copy(); d2.loc[d2["youtube_spend_imputed"] == 1, "youtube_spend"] *= 1.25
sens["YouTube filled-in weeks +25%"] = marginal_roi(*fit(wins["all weeks"], df=d2)[:2])
print(pd.DataFrame(sens).round(2))

# ===== 4. Meta geo-holdout experiment (NOT used to fit the model) =====
print("\n4. META EXPERIMENT")
lift = pd.read_csv("data/lift_test_results.csv")
print(lift[["holdout_share_of_spend", "spend_withheld_inr", "incremental_revenue_inr", "measured_iroas", "iroas_90ci_low", "iroas_90ci_high"]].to_string(index=False))
test_wk = d["week_start"].between("2025-04-07", "2025-05-12")          # 6 test weeks
window = d["week_start"].between("2025-04-07", "2025-06-09")            # + 4 post weeks
def model_iroas(model, scale, extra, decay=.3, saturate=True):
    d_cf = d.copy(); d_cf.loc[test_wk, "meta_spend"] *= 0.8             # switch off 20% of Meta in test weeks
    p_base = model.predict(sm.add_constant(make_X(d, scale, decay, saturate, extra)))
    p_cf = model.predict(sm.add_constant(make_X(d_cf, scale, decay, saturate, extra)))
    lost = (p_base - p_cf)[window].sum()
    return lost / (0.2 * d.loc[test_wk, "meta_spend"].sum())
iroas = model_iroas(mf, scale, ctrl)
lo, hi = mf.conf_int().loc["meta_spend"] / mf.params["meta_spend"] * iroas      # loss is proportional to the Meta coefficient
print("final model, same 20% cut:  iROAS", round(iroas, 2), " 95% range", round(lo, 2), "to", round(hi, 2))
mb, sb, _ = fit(d["t"] >= 0, extra=ctrl + ["brand_search_index"])
print("final + brand search:       iROAS", round(model_iroas(mb, sb, ctrl + ["brand_search_index"]), 2))
m_raw = sm.OLS(d["revenue_inr"], sm.add_constant(d[channels + v0_extra])).fit()
print("v0 on cleaned data:         ", round(m_raw.params["meta_spend"], 2))
print("v0 as the agency ran it:     0.40  (earlier result on the raw file)")
print("Meta spend in the test weeks vs 2025 average:",
      round(d.loc[test_wk, "meta_spend"].mean() / d.loc[d["week_start"] >= "2025-01-01", "meta_spend"].mean(), 2), "x")
