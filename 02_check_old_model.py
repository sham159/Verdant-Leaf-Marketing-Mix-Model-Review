# Step 2 - what is wrong with the v0 model?
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

channels = ["meta_spend", "google_search_spend", "youtube_spend",
            "influencer_spend", "tv_spend", "amazon_ads_spend"]
controls = ["festive_flag", "avg_discount_pct"]

def fit(data, xcols):
    return sm.OLS(data["revenue_inr"], sm.add_constant(data[xcols])).fit()

# 1. v0 exactly as the agency ran it (raw file, dropna)
raw = pd.read_csv("data/mmm_weekly_data.csv")
v0 = fit(raw.dropna(), channels + controls)
print("v0 on raw: rows =", len(raw.dropna()), " R2 =", round(v0.rsquared, 3))
print(v0.params[channels].round(2))

# 2. same spec on cleaned data (dedup, lakh fix, tv credit, youtube filled)
clean = pd.read_csv("data/mmm_weekly_clean.csv", parse_dates=["week_start"])
v0c = fit(clean, channels + controls)
print("\nsame spec on clean data: rows =", len(clean), " R2 =", round(v0c.rsquared, 3))
print(v0c.params[channels].round(2))

# 3. add the other demand/promo variables
more = ["marketplace_sale_flag", "competitor_promo_flag", "new_launch_flag", "price_index"]
m2 = fit(clean, channels + controls + more)
print("\n+ promo/price controls: R2 =", round(m2.rsquared, 3))
print(m2.params[channels + more].round(2))

# 4. multicollinearity
X = sm.add_constant(clean[channels + controls])
print("\nVIF:")
for i, c in enumerate(X.columns[1:], start=1):
    print(f"  {c:<22}{variance_inflation_factor(X.values, i):6.1f}")
print("\nspend correlations with festive / discount:")
print(clean[channels + ["festive_flag", "avg_discount_pct"]].corr().loc[channels, ["festive_flag", "avg_discount_pct"]].round(2))

# 5. endogeneity clue: does spend follow demand from the week before?
clean["brand_search_prev"] = clean["brand_search_index"].shift(1)
clean["organic_prev"] = clean["organic_traffic_index"].shift(1)
clean["rev_prev"] = clean["revenue_inr"].shift(1)
print("\ncorrelation of this week's spend with LAST week's demand:")
print(clean[channels + ["brand_search_prev", "organic_prev", "rev_prev"]].corr().loc[channels, ["brand_search_prev", "organic_prev", "rev_prev"]].round(2))
print("google_search_spend vs brand_search_index same week:",
      round(clean["google_search_spend"].corr(clean["brand_search_index"]), 2))

# 6. in-sample vs out-of-sample (train 2023-24, test 2025)
train = clean[clean["week_start"] < "2025-01-01"]
test = clean[clean["week_start"] >= "2025-01-01"]
cols = channels + controls
mt = fit(train, cols)
pred = mt.predict(sm.add_constant(test[cols], has_constant="add"))
sse = ((test["revenue_inr"] - pred) ** 2).sum()
sst = ((test["revenue_inr"] - train["revenue_inr"].mean()) ** 2).sum()
print("\nv0 spec: train R2 =", round(mt.rsquared, 3), " test R2 (2025) =", round(1 - sse / sst, 3),
      " test MAPE =", round(((test['revenue_inr'] - pred).abs() / test['revenue_inr']).mean() * 100, 1), "%")
print("train-only coefficients:"); print(mt.params[channels].round(2))

# 7. carryover: does last week's spend help explain this week's revenue?
lag = clean.copy()
for c in channels:
    lag[c + "_lag1"] = lag[c].shift(1)
lag = lag.dropna()
lagcols = [c + "_lag1" for c in channels]
ml = fit(lag, channels + lagcols + controls + more)
print("\nwith 1-week lag of each channel (+ controls): R2 =", round(ml.rsquared, 3))
print(pd.DataFrame({"same_week": ml.params[channels].values, "lag1": ml.params[lagcols].values,
                    "lag1_pvalue": ml.pvalues[lagcols].values}, index=channels).round(2))

# 8. saturation clue: revenue response at high vs low spend (Meta, Google)
for c in ["meta_spend", "google_search_spend", "amazon_ads_spend"]:
    q = pd.qcut(clean[c], 4, labels=False)
    print(f"\n{c} quartile -> avg revenue (Cr), avg spend (Cr):")
    print(clean.groupby(q).agg(spend=(c, "mean"), rev=("revenue_inr", "mean")).div(1e7).round(2).T)

# 9. what dropna dropped
print("\nweeks dropped by dropna:", raw[raw.isna().any(axis=1)]["week_start"].tolist())
print("Meta experiment iROAS = 1.39 (90% CI 0.94-1.84); v0 Meta coef =", round(v0.params['meta_spend'], 2))
