# Step 1 - data audit and cleaning for Verdant Leaf MMM
import pandas as pd

raw = pd.read_csv("data/mmm_weekly_data.csv")   # raw file is never modified
df = raw.copy()

spend_cols = ["meta_spend", "google_search_spend", "youtube_spend",
              "influencer_spend", "tv_spend", "amazon_ads_spend"]

# ---------- 1. basic look ----------
print(df.shape)
print(df.head())
print(df.tail())
print(df.columns.tolist())
df.info()
print(df.describe().T)
print(df.isna().sum()[df.isna().sum() > 0])

# ---------- 2. duplicates and dates ----------
print("duplicate rows:", df.duplicated().sum())
print("unique weeks:", df["week_start"].nunique())
print(df[df["week_start"].duplicated(keep=False)])      # are the two copies identical?

df["week_start"] = pd.to_datetime(df["week_start"])
df = df.drop_duplicates().reset_index(drop=True)
print("rows after dedup:", len(df))
print("all Mondays:", (df["week_start"].dt.dayofweek == 0).all())
print("gaps between weeks:", df["week_start"].diff().value_counts().to_dict())

# ---------- 3. meta_spend scale ----------
# impressions look normal in the odd weeks, so check cost per 1000 impressions
df["meta_cpm"] = df["meta_spend"] / df["meta_impressions"] * 1000
low = df["meta_spend"] < 1000
print("tiny meta_spend weeks:", low.sum(), df.loc[low, "week_start"].min().date(),
      "to", df.loc[low, "week_start"].max().date())
print("CPM normal weeks:", df.loc[~low, "meta_cpm"].median().round(1))
print("CPM odd weeks x 100,000:", (df.loc[low, "meta_cpm"] * 1e5).median().round(1))
# odd weeks are in lakhs (1 lakh = 100,000) -> convert to rupees
df.loc[low, "meta_spend"] = df.loc[low, "meta_spend"] * 100000

# ---------- 4. negative tv_spend ----------
print((df[spend_cols] < 0).sum()[lambda s: s > 0])
print(df.loc[df["week_start"].between("2024-10-07", "2024-11-25"),
             ["week_start", "tv_spend", "revenue_inr", "festive_flag"]])
# single round-number negative, one week after the festive TV burst ended, no TV airing -> credit/reversal
df["tv_credit_flag"] = (df["tv_spend"] < 0).astype(int)
df.loc[df["tv_spend"] < 0, "tv_spend"] = 0

# ---------- 5. missing youtube_spend ----------
print(df.loc[df["youtube_spend"].isna(), ["week_start", "youtube_spend", "youtube_impressions"]])
print("weeks with zero spend and zero impressions:",
      ((df["youtube_spend"] == 0) & (df["youtube_impressions"] == 0)).sum(),
      "| zero spend but impressions > 0:",
      ((df["youtube_spend"] == 0) & (df["youtube_impressions"] > 0)).sum())
df["yt_cpm"] = df["youtube_spend"] / df["youtube_impressions"] * 1000
print(df.groupby(df["week_start"].dt.year)["yt_cpm"].agg(["median", "std"]))
# the 4 missing weeks have impressions, so ads did run -> estimate spend from impressions
df["youtube_spend_imputed"] = df["youtube_spend"].isna().astype(int)
df["youtube_spend"] = df["youtube_spend"].fillna(df["youtube_impressions"] * df["yt_cpm"].median() / 1000)

# ---------- 6. other checks (no changes made) ----------
print(df.filter(like="flag").apply(lambda s: s.unique().tolist()))
print("AOV range:", (df["revenue_inr"] / df["orders"]).agg(["min", "max"]).round(0).tolist())
print(df.groupby(df["week_start"].dt.year)[spend_cols].sum().div(1e7).round(2))   # in crore
print("2025 total spend (Cr):", round(df[df["week_start"].dt.year == 2025][spend_cols].sum().sum() / 1e7, 2))

# check against the Meta experiment file: 20% of spend during the 6 test weeks
test_weeks = df["week_start"].between("2025-04-07", "2025-05-12")
print("test weeks:", test_weeks.sum(), "| 20% of Meta spend:", round(0.2 * df.loc[test_weeks, "meta_spend"].sum()),
      "| lift file says 4,459,000")

# ---------- save ----------
df = df.drop(columns=["meta_cpm", "yt_cpm"])
df["meta_unit_fixed_flag"] = low.astype(int)
df.to_csv("data/mmm_weekly_clean.csv", index=False)
print("saved", df.shape)
