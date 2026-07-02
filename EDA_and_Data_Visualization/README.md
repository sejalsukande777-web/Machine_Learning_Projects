# Diabetes Prediction

Predicts whether a patient is diabetic based on medical diagnostic measurements, using Exploratory Data Analysis followed by classification models.

## Dataset

Pima Indians Diabetes Dataset — 768 records with the following columns:

- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target: 0 = Not Diabetic, 1 = Diabetic)

## Approach

1. **EDA** — load dataset, check nulls, view stats with `.describe()`, plot class distribution, feature histograms, and boxplots to spot outliers.
2. **Preprocessing** — replace invalid zero values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI with column mean; scale features with `StandardScaler`.
3. **Model Building** — train Logistic Regression, K-Nearest Neighbors, and Decision Tree classifiers.
4. **Evaluation** — compare accuracy, precision, recall, F1 score, and confusion matrix for each model.
5. **Output** — predict on test data and save results to `diabetes_predictions.csv`.

## How to run

```bash
python diabetes_prediction.py
```

## Output

- `outcome_distribution.png`
- `feature_histograms.png`
- `feature_boxplot.png`
- confusion matrix plots per model
- `diabetes_predictions.csv`