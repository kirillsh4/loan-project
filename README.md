# Loan Default Classifier

Reproducible RandomForest pipeline for predicting loan default risk.

## What the model does

Trains a binary classifier that predicts whether a loan applicant will default,
using applicant demographics, financial features, and credit history. The model
is a `RandomForestClassifier` wrapped in a scikit-learn `Pipeline` with imputation,
ordinal encoding for education, and one-hot encoding for categoricals.
Hyperparameters are tuned with `RandomizedSearchCV` over a stratified 5-fold CV.

## Repo layout

```
loan-default-classifier/
├── README.md           
├── requirements.txt    — pinned training dependencies
├── .gitignore
├── modelling.ipynb     — EDA & exploration
├── train.py            — single-command reproducible training pipeline
└── data/
    └── .gitkeep        
```

## Results & findings

Held-out test scores from the best CV-selected model:

- F1: **0.82** · ROC-AUC: **0.97** · PR-AUC: **0.92** · Precision: **0.90** · Recall: **0.76**

Top features: (features which affect whether a customer will default a loan or not)

- **`loan_percent_income`** : loan size relative to income. Higher ratios mean the repayment burden eats more of the applicant's earnings, so default risk goes up.
- **`previous_loan_defaults_on_file`** : People who have defaulted before are more likely to default again
- **`loan_int_rate`** : Higher rates both reflect the lender's prior risk assessment and make repayment harder, so it correlates strongly with default.
- **`person_income`** : Higher income gives more cushion against shocks and therefore likely to lower default probability.

## Quickstart

```
git clone <repo-url>
cd loan-default-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place loan_data_raw.csv in data/
python train.py
```

Outputs after a successful run:
- `model.joblib` — the fitted best estimator
- `metrics.json` — best params, CV F1, test metrics, top-10 feature importances

### In progress features/notes

- Currently working on having the model compute probability output, not just predictions. So the model can be used for scoring / expected-loss calculations.

## License

MIT 
