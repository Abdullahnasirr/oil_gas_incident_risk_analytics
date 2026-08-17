# Oil & Gas Pipeline Incident Risk Analytics

Analyzed 2,024 Canadian pipeline incidents (2008–2026) from the Canada Energy Regulator to identify risk factors associated with significant incidents and built a classification model to predict incident severity.

## Results
- **XGBoost classifier achieved ROC-AUC of 0.926** on held-out test data
- **67% recall on significant incidents** with 90% overall accuracy
- **Emergency Level, Volume Released, and Substance Type** were the strongest predictors of incident significance
- Baseline logistic regression only caught 5% of significant incidents showing why accuracy alone isnt enough data for imbalanced safety datasets

## Project Structure

```
oil-gas-incident-risk-analytics/
├── data/
│   ├── pipeline-incidents-comprehensive-data.csv    # Raw CER data
│   └── pipeline_incidents_clean.csv                 # Cleaned dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb                    # EDA
│   ├── 02_visualizations.ipynb                      # Charts
│   └── 03_ml_model.ipynb                            # ML models
├── models/
│   └── xgb_model_final.pkl                          # Saved XGBoost model
└── dashboard/                                        # Exported charts
```

## Dataset
**Source:** [Canada Energy Regulator — Pipeline Incident Data](https://open.canada.ca/data/en/dataset/7dffedc4-23fa-440c-a36d-adf5a6cc09f1)

- 2,024 incidents from federally regulated pipelines (2008–2026)
- Updated quarterly by the CER
- 102 raw features reduced to 15 engineered features

## Methodology

### 1. Data Cleaning
- Selected 15 relevant features from 102 raw columns
- Converted target variable (Significant: Yes/No) to binary (1/0)
- Handled mixed-type volume column containing both numeric values and "Not Applicable" strings
- Filled missing categorical values

### 2. Exploratory Analysis
- 2017 was the peak year for incidents (~175); general declining trend since
- External Interference is the #1 cause (345 incidents)
- Alberta accounts for ~35% of all incidents
- 18% of incidents classified as Significant — class imbalance addressed in modelling

### 3. Machine Learning
Three models trained and evaluated with class imbalance handling:

| Model | Test Accuracy | Significant Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 84.9% | 0.05 | — |
| Random Forest | 89.6% | 0.48 | — |
| XGBoost (tuned) | 90.0% | 0.67 | 0.926 |

**Class imbalance strategy:** `scale_pos_weight` in XGBoost and `class_weight='balanced'` in Random Forest to penalize missed significant incidents more heavily.

### 4. Key Insight — False Negatives
In a safety context, false negatives (predicting not significant when it actually is) are far more costly than false positives. The model missed 20 significant incidents out of 61 in the test set. Optimizing for recall over accuracy was a deliberate design choice.

## Tech Stack
- **Python** — pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
- **Machine Learning** — Logistic Regression, Random Forest, XGBoost
- **Visualization** — matplotlib, seaborn, Power BI
- **Tools** — Jupyter Notebook, PyCharm, Git/GitHub

## How to Run
1. Clone the repo
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter`
3. Download the raw data from the CER link above and place in `/data`
4. Run notebooks in order: 01 → 02 → 03

## Author
Abdullah Nasir — 2nd year BSc Data Science student, University of Calgary