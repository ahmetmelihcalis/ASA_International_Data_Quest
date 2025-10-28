# ASA International Data Quest Competition Submission

## Luxury Data Privilege Analysis

**Focus:** Measuring Global Data Infrastructure Inequality

---

## Information

This repository examines the concept of "Luxury Data Privilege" (systematic inequality in data accessibility and quality among countries). Using the World Bank Development Indicators (WDI), it analyzes whether high-income countries possess significantly richer datasets.

---

## Research Question

**Do wealthier nations enjoy privileged access to comprehensive, high-quality data infrastructure?**

This project examines whether the concept of "luxury indicators"—metrics such as patent applications, venture capital flows, R&D expenditure, and high-technology exports—are systematically underrepresented in lower-income countries, thereby creating a feedback loop of invisibility in global research and policy discourse.

---

## Methodology

### Data Sources:

All analysis is based on publicly available World Bank data:

* `WDICSV.csv` — Core time series data
* `WDICountry.csv` — Country metadata (income groups, regions)
* `WDISeries.csv` — Indicator descriptions and classifications

### Indicator Selection:

A keyword-based approach is used to identify "luxury indicators":

```
patent applications, trademark applications, industrial design,
R&D expenditure, researchers in R&D, scientific journals,
market capitalization, stock market metrics, venture capital,
high-technology exports, ICT goods exports, broadband subscriptions,
internet usage
```

If keyword matching yields fewer than the indicator cap (default: 300), the list is augmented with the most frequently reported indicators in the dataset.

### Scoring Algorithm:

For each country, we calculate:

1.  **Coverage (%)** = Proportion of selected luxury indicators available
2.  **Data Completeness (%)** = Proportion of non-missing values (2003–2023)
3.  **Indicator Count** = Absolute number of available indicators

**Base Score =** Coverage × Fill Penalty × Count Penalty

Where:
* **Fill Penalty** ∈ [0.3, 1.0] — penalizes incomplete time series
* **Count Penalty** ∈ [0.4, 1.0] — penalizes limited indicator availability

### Dual Scoring Framework:

The analysis is conducted using two different scenarios:

**1. NEUTRAL SCORING (Main Model)**
```
Multiplier = 1.0 for all countries
```
Provides an objective measurement of data infrastructure quality without income-based adjustments. This approach aims to reveal the *natural* correlation (or lack thereof) between economic development and data privilege.

**2. ADJUSTED SCORING (Sensitivity Analysis)**
```
High income:          1.30
Upper middle income:  1.12
Lower middle income:  0.92
Low income:           0.78
```


This model is **not** a robustness test. Its purpose is to run a 'what-if scenario' to see how the statistical differences between groups would change if an *intentional* income-based bias were added to the model.

---

## Statistical Framework

### Hypothesis Testing

**Null Hypothesis (H₀):** Mean data privilege scores are equal across income groups.

**Alternative Hypothesis (H₁):** At least one income group's mean score differs significantly.

**Method:** One-way ANOVA followed by Tukey HSD post-hoc test for pairwise comparisons.

## Key Parameters

The following parameters can be modified at the top of the analysis script:

| Parameter | Default | Description |
|---|---|---|
| `indicator_cap` | 300 | Maximum number of indicators to include |
| `use_offset` | False | Apply a deterministic offset to break potential ties |
| `random_seed` | 0 | Seed for reproducibility |
| `fill_penalty_min` | 0.3 | Minimum penalty for incomplete data |
| `count_penalty_min` | 0.4 | Minimum penalty for low indicator count |

---

## Output Files

### Core Results

**NEUTRAL SCORING (Main Model)**
* `luxury_scores_neutral.csv` — Complete country rankings
* `anova_neutral.txt` — Statistical test results
* `posthoc_neutral.txt` / `.csv` — Pairwise comparison results
* `tukey_plot_neutral.png` — Visualization with confidence intervals

**ADJUSTED SCORING (Sensitivity Analysis)**
* `luxury_scores_adjusted.csv` — Income-weighted rankings
* `anova_adjusted.txt` — Statistical test results
* `posthoc_adjusted.txt` / `.csv` — Pairwise comparison results
* `tukey_plot_adjusted.png` — Visualization with confidence intervals

### Interactive Visualizations

All HTML files in `outputs/interactive/` provide:
* Choropleth maps with hover details
* Coverage and data completeness breakdowns
* Income group classifications
* Responsive color scales optimized for data distribution

**Continuous Scale Maps:** Shows the full spectrum of scores.
**Binned Maps:** Quintile-based categorical visualization.

---

## Limitations

1.  **Indicator Selection:** The keyword-based approach may miss relevant indicators or include tangential ones.
2.  **Temporal Scope:** The 20-year window may not capture longer-term trends.
3.  **Causality:** This analysis identifies correlation, not causation, between income and data quality.
4.  **Aggregation:** Country-level analysis masks within-country regional disparities.
5.  **Threshold Effects:** The binary (present/absent) check for indicators does not capture gradations in data quality.

---

## Result

In this competition, I aimed to investigate whether a meaningful relationship exists between countries’ development levels (assuming high-income countries are more developed) and their data gaps and data quality, using datasets from the World Development Indicators (WDI).

To validate this, I applied ANOVA and Tukey-HSD tests across two different scenarios:

### 1. Main Finding (Neutral Scoring)

The analysis of the 'Neutral Scoring' model **confirms the "Luxury Data Privilege" hypothesis** in a nuanced way. The ANOVA test confirmed a significant difference exists between groups (rejecting H₀), and the Tukey HSD post-hoc test identified precisely where those differences lie.

**The key finding is that a significant data privilege gap exists between High-Income countries and Middle-Income countries.**

* `High Income` scores are statistically **higher** than `Lower Middle Income` scores.
* `High Income` scores are also statistically **higher** than `Upper Middle Income` scores

**However, in a crucial and interesting finding, the data gap between `High Income` and `Low Income` countries was *not* found to be statistically significant**

This suggests the relationship between wealth and data infrastructure is not linear. The "data privilege gap" is most pronounced between the wealthiest nations and the middle-income bloc, rather than a simple split between the "rich" and the "poor".

### 2. Sensitivity Analysis (Adjusted Scoring)

The 'Adjusted Scoring' scenario, which applies artificial income-based multipliers, serves as a "what-if" analysis. This model attempts to amplify the differences observed in the neutral model by applying a `1.30` multiplier to high-income countries and a `0.78` multiplier to low-income countries. As expected, this manipulation increases the statistical gaps between groups, confirming the model's sensitivity to income-based weighting.

Visualizations and maps illustrating both scenarios can be found in the `outputs/` directory. The `posthoc_neutral.csv` file contains the definitive statistical results for the main finding.