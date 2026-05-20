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

Train Scores from the best CV-selected model:

- F1: **0.82** · Precision: **0.89** · Recall: **0.75**

Test Scores from the best CV-selected model:

- F1: **0.82** · Precision: **0.90** · Recall: **0.76**

Both the train and test scores are very similar meaning it is safe to assume the model generalises quite well, and has a good bias-variance trade-off.

The imbalanced split between loan defaulters and non-defaulters matters for how precision and recall are evaluated. And which metric is more important is dependant on the purpose of the model/business objective.

For example, if the primary business goal is to avoid losses from defaults, recall is the important metric as it measures false-negatives. In this case (0.76 recall), depending on the cost of a default, the decision threshold may need to be shifted downward to trade some precision for more recall. 

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

- Currently working on having the model compute probability output, not just predictions. So the model can be used for scoring / expected-loss calculations. Next time I could use a confusion-matrix after training to better evaluate model.

## License

MIT 
