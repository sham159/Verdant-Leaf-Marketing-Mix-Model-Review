# Verdant Leaf: Marketing Mix Model Review

Take-home assignment. I had to find out what is wrong with the agency's marketing model, build a better one, and decide how to split the ₹16 Cr budget for Jan–Mar 2026. All data is synthetic.

**Report (3 pages):** `Verdant_Leaf_MMM_Report.pdf`

## Files

| File | What it does |
|---|---|
| `01_clean_data.py` | Checks the raw data and fixes the problems. Saves `data/mmm_weekly_clean.csv` |
| `02_check_old_model.py` | Runs the agency's old model (V0) and checks what is wrong with it |
| `03_better_model.py` | The better model (other drivers, carryover, each extra rupee earns a bit less) |
| `04_validate_model.py` | Tests the model on weeks it has not seen, and compares it with the Meta test |
| `05_budget_split.py` | Splits the ₹16 Cr budget. Saves `budget_recommendation.csv` |
| `data/` | Raw files from the assignment, plus the cleaned file |

## How to run

```
pip install -r requirements.txt
python 01_clean_data.py
python 02_check_old_model.py
python 03_better_model.py
python 04_validate_model.py
python 05_budget_split.py
```

Run them from the main folder, in this order. The raw data files are never changed.

## Short summary

**Data problems:** one duplicate week, Meta spend in lakhs for 13 weeks in late 2023, one negative TV value, 4 weeks of missing YouTube spend.

**Old model:** the data errors made Meta look useless (0.40). Google looked like 13.7 because we spend more on Google when people are already searching for the brand. It also ignored sales, promos and launches, and carryover.

**Better model:** a simple regression with festive, sale, promo and launch weeks, a growth trend, carryover (0.3) and diminishing returns. Its Meta result (1.35) is close to the Meta test (1.39).

**Budget (₹ Cr per quarter):**

| | Meta | Google | YouTube | Influencer | TV | Amazon |
|---|---|---|---|---|---|---|
| Now | 4.93 | 2.31 | 0.96 | 1.19 | 2.45 | 4.16 |
| Recommended | 5.69 | 2.77 | 0.96 | 1.19 | 1.23 | 4.16 |

Expected gain is about ₹2 to ₹4 Cr of extra revenue in the quarter. It could be less if Google's return is mostly sales we would get anyway.

## Assumptions I made

- The 13 small Meta values are in lakhs. This is a guess from the pattern (cost per 1,000 impressions), the file does not say.
- The negative TV value is a credit, so I set it to 0.
- The 4 missing YouTube weeks are estimated from impressions and the usual cost per 1,000.
- "2025 average quarterly spend" = the 51 weeks starting in 2025, divided by 4 (₹64 Cr a year, as in the brief).
- Carryover of 0.3 and the "each extra rupee earns less" shape are my choices. The data does not prove them.

## Tools used

Claude for writing and debugging the Python code, and ChatGPT for understanding concepts. I checked the results by running the scripts.
