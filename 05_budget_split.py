# Step 5 - how to split the Rs 16 Cr for Jan-Mar 2026
import pandas as pd
import numpy as np
import statsmodels.api as sm

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
d = pd.read_csv("data/mmm_weekly_clean.csv", parse_dates=["week_start"])
d["t"] = np.arange(len(d))
channels = ["meta_spend", "google_search_spend", "youtube_spend",
            "influencer_spend", "tv_spend", "amazon_ads_spend"]
ctrl = ["festive_flag", "marketplace_sale_flag", "competitor_promo_flag", "new_launch_flag", "t"]
decay = .3
TOTAL = 16e7                      # Rs 16 Cr
WEEKS = 13

# ---- 1. 2025 average quarterly spend (the baseline for the 50-150% rule) ----
# 2025 = the 51 weeks that start in 2025 (the data stops at 2025-12-22); this gives Rs 64 Cr a year as in the brief
base = d[d["week_start"].dt.year == 2025][channels].sum() / 4
last52 = d.tail(52)               # alternative: last 52 weeks, only used as a check
print("2025 quarterly baseline (Rs Cr)")
print(pd.DataFrame({"used (2025 weeks)": base / 1e7, "alt: last 52 weeks": last52[channels].sum() / 4 / 1e7}).round(2))
print("total:", round(base.sum() / 1e7, 2), "Cr per quarter (", round(base.sum() * 4 / 1e7, 1), "Cr a year )")

# ---- 2. response curves from the Step 3 model ----
scale = {c: d[c].mean() for c in channels}
def fit(extra):
    X = pd.DataFrame({c: np.log1p(np.array([0.0] * len(d)) + 0) for c in channels})
    for c in channels:
        a, prev = [], 0
        for v in d[c].values:
            prev = v + decay * prev
            a.append(prev)
        X[c] = np.log1p(np.array(a) / scale[c])
    X = pd.concat([X, d[extra]], axis=1)
    return sm.OLS(d["revenue_inr"], sm.add_constant(X)).fit()

mA = fit(ctrl)                                       # final model
mB = fit(ctrl + ["brand_search_index"])              # cautious version (Google much lower)
bA, bB = mA.params[channels], mB.params[channels]
bMix = (bA + bB) / 2                                 # best guess when we cannot tell A from B
print("\nchannel coefficients (Rs Cr per week per unit of log response)")
print(pd.DataFrame({"A_final": bA / 1e7, "B_brand_search": bB / 1e7}).round(2))

def response(c, quarter_spend):
    weekly = quarter_spend / WEEKS
    return np.log1p(weekly / ((1 - decay) * scale[c]))          # steady-state carryover

def gain(alloc, b):
    # extra revenue over the quarter vs the current split
    return sum(WEEKS * b[c] * (response(c, alloc[c]) - response(c, base[c])) for c in channels)

def marginal(c, spend, b):
    return b[c] / ((1 - decay) * scale[c] + spend / WEEKS)       # extra revenue per extra Rs 1

# ---- 3. allocate: add Rs 1 lakh at a time to the channel with the best marginal return ----
def allocate(b, lo, hi):
    alloc = {c: lo[c] for c in channels}
    left = TOTAL - sum(alloc.values())
    while left > 1:
        open_ch = [c for c in channels if alloc[c] + 1e5 <= hi[c]]
        best = max(open_ch, key=lambda c: marginal(c, alloc[c], b))
        alloc[best] += 1e5
        left -= 1e5
    alloc[best] += left                                          # rounding leftover
    return alloc

full_lo = {c: .5 * base[c] for c in channels}
full_hi = {c: 1.5 * base[c] for c in channels}

# a) the v0 rule: straight-line ROI, shift money to the highest coefficient
v0 = sm.OLS(d["revenue_inr"], sm.add_constant(d[channels + ["festive_flag", "avg_discount_pct"]])).fit().params[channels]
v0_alloc = {c: full_lo[c] for c in channels}
left = TOTAL - sum(v0_alloc.values())
for c in v0.sort_values(ascending=False).index:
    add = min(full_hi[c] - v0_alloc[c], left)
    v0_alloc[c] += add; left -= add

# b) model-based, full 50-150% freedom, using the final model
free_alloc = allocate(bA, full_lo, full_hi)

# c) optimizer with guardrails (cross-check): hold YouTube/Amazon flat, Google at most +-20%
lo, hi = dict(full_lo), dict(full_hi)
for c in ["youtube_spend", "amazon_ads_spend"]:      # range spans zero -> hold flat
    lo[c] = hi[c] = base[c]
lo["google_search_spend"], hi["google_search_spend"] = .8 * base["google_search_spend"], 1.2 * base["google_search_spend"]
guarded = allocate(bMix, lo, hi)

# d) recommended: simple rules, so every move can be explained
rec = dict(base)
rec["tv_spend"] = .5 * base["tv_spend"]                    # only channel whose return is clearly below Rs 1
rec["google_search_spend"] = 1.2 * base["google_search_spend"]   # best estimate, but likely inflated -> cap at +20%
rec["meta_spend"] = TOTAL - sum(v for k, v in rec.items() if k != "meta_spend")   # remainder; experiment-backed channel
# youtube, influencer, amazon stay at the 2025 level

options = {"current": dict(base), "v0 rule (shift to highest ROI)": v0_alloc,
           "model, no guardrails": free_alloc, "optimizer + guardrails": guarded, "recommended": rec}
print("\nallocations (Rs Cr) and % of 2025 baseline")
tab = pd.DataFrame({k: pd.Series(v) / 1e7 for k, v in options.items()}).round(2)
print(tab); print("totals:", tab.sum().round(2).to_dict())
print((pd.DataFrame(options).div(base, axis=0) * 100).round(0).drop(columns="current"))

# ---- 4. what do we expect? (extra revenue per quarter vs current split, Rs Cr) ----
cov = mA.cov_params().loc[channels, channels].values
draws = np.random.default_rng(0).multivariate_normal(bA.values, cov, 5000)
print("\nexpected extra revenue per quarter (Rs Cr) under different assumptions")
rows = {}
for name, a in options.items():
    if name == "current": continue
    g = np.array([gain(a, pd.Series(dr, index=channels)) for dr in draws]) / 1e7
    rows[name] = {"model A": gain(a, bA) / 1e7, "model B (brand search)": gain(a, bB) / 1e7,
                  "A, 5th pct": np.percentile(g, 5), "A, 95th pct": np.percentile(g, 95),
                  "share of draws > 0": (g > 0).mean()}
print(pd.DataFrame(rows).T.round(2))
print("typical quarterly revenue (last 52 weeks):", round(last52["revenue_inr"].sum() / 4 / 1e7, 1), "Cr")

# ---- 5. marginal return at the recommended split ----
print("\nmarginal ROI at recommended spend: model A | model B")
for c in channels:
    print(f"  {c:<22}{marginal(c, rec[c], bA):6.2f} | {marginal(c, rec[c], bB):6.2f}  (2025 spend level: {marginal(c, base[c], bA):5.2f} | {marginal(c, base[c], bB):5.2f})")

# the guardrail moves are what matter: one-by-one effect of each recommended change (model A | model B)
print("\nvalue of each move on its own (Rs Cr per quarter): model A | model B")
for c, new in [("tv_spend", rec["tv_spend"]), ("google_search_spend", rec["google_search_spend"]), ("meta_spend", rec["meta_spend"])]:
    one = {k: base[k] for k in channels}; one[c] = new
    print(f"  {c:<22}{gain(one, bA)/1e7:6.2f} | {gain(one, bB)/1e7:6.2f}   (spend change {(new-base[c])/1e7:+.2f} Cr)")

# ---- save the recommendation for the report ----
out = pd.DataFrame({"current_cr": base / 1e7, "recommended_cr": pd.Series(rec) / 1e7})
out["change_cr"] = out["recommended_cr"] - out["current_cr"]
out["pct_of_2025"] = out["recommended_cr"] / out["current_cr"] * 100
out["marginal_roi_A"] = [marginal(c, rec[c], bA) for c in channels]
out["marginal_roi_B"] = [marginal(c, rec[c], bB) for c in channels]
out.round(3).to_csv("budget_recommendation.csv")
print("\nrecommended total (Cr):", round(out["recommended_cr"].sum(), 4), "| all within 50-150%:",
      bool(out["pct_of_2025"].between(50, 150).all()))
print(out.round(2))
