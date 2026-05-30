import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score, f1_score,
                             classification_report)


def DiabetesPrediction(datapath):
    border = "-" * 50

    # ------------------------------------------------
    # Step 1 : Load Dataset
    # ------------------------------------------------
    print(border)
    print("Step 1 : Load Dataset")
    print(border)

    df = pd.read_csv(datapath)
    print("First 5 records from the dataset :")
    print(df.head())

    target_column = None
    if 'Actual Outcome' in df.columns:
        target_column = 'Actual Outcome'
    elif 'Outcome' in df.columns:
        target_column = 'Outcome'
    else:
        raise ValueError("Dataset must contain either 'Actual Outcome' or 'Outcome' column.")

    numeric_df = df.select_dtypes(include='number')
    print(f"Using target column: '{target_column}'")

    # ------------------------------------------------
    # Step 2 : Remove Unwanted Columns
    # ------------------------------------------------
    print(border)
    print("Step 2 : Remove Unwanted Columns")
    print(border)

    print("Shape of dataset before removal :", df.shape)
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    print("Shape of dataset after removal  :", df.shape)

    print("\nClean dataset :")
    print(df.head())

    # ------------------------------------------------
    # Step 3 : Check Missing Values
    # ------------------------------------------------
    print(border)
    print("Step 3 : Check Missing Values")
    print(border)

    print("Missing values count :\n", df.isnull().sum())

    # ------------------------------------------------
    # Step 4 : Display Column Info
    # ------------------------------------------------
    print(border)
    print("Step 4 : Display Column Info")
    print(border)

    df.info()

    # ------------------------------------------------
    # Step 5 : Display Statistical Summary
    # ------------------------------------------------
    print(border)
    print("Step 5 : Display Statistical Summary")
    print(border)

    print(df.describe())

    # ------------------------------------------------
    # Step 6 : EDA - Plot Target Variable Distribution
    # ------------------------------------------------
    print(border)
    print("Step 6 : EDA - Plot Target Variable Distribution")
    print(border)

    print(f"{target_column} value counts :\n", df[target_column].value_counts())

    plt.figure(figsize=(6, 4))
    df[target_column].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
    plt.title("Distribution of Outcome (0 = Non-Diabetic, 1 = Diabetic)")
    plt.xlabel(target_column)
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outcome_distribution.png")
    plt.close()
    print("Plot saved as outcome_distribution.png")

    # ------------------------------------------------
    # Step 7 : EDA - Histogram of All Features
    # ------------------------------------------------
    print(border)
    print("Step 7 : EDA - Histogram of All Features")
    print(border)

    if numeric_df.shape[1] > 0:
        numeric_df.hist(figsize=(12, 8), bins=20, color='steelblue', edgecolor='black')
        plt.suptitle("Histogram of Numeric Features", y=1.02)
        plt.tight_layout()
        plt.savefig("histograms.png")
        plt.close()
        print("Plot saved as histograms.png")
    else:
        print("No numeric columns available for histograms.")

    # ------------------------------------------------
    # Step 8 : EDA - Boxplot to Detect Outliers
    # ------------------------------------------------
    print(border)
    print("Step 8 : EDA - Boxplot to Detect Outliers")
    print(border)

    boxplot_df = numeric_df.drop(columns=[target_column], errors='ignore')
    if boxplot_df.shape[1] > 0:
        plt.figure(figsize=(14, 6))
        boxplot_df.boxplot()
        plt.title("Boxplot of Features (to Detect Outliers)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("boxplot.png")
        plt.close()
        print("Plot saved as boxplot.png")
    else:
        print("No numeric feature columns available for boxplots.")

    # ------------------------------------------------
    # Step 9 : EDA - Correlation Matrix
    # ------------------------------------------------
    print(border)
    print("Step 9 : EDA - Correlation Matrix")
    print(border)

    corr_df = numeric_df.corr()
    print("Correlation Matrix :\n")
    print(corr_df)

    if corr_df.shape[0] > 1:
        plt.figure(figsize=(10, 7))
        sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("correlation_heatmap.png")
        plt.close()
        print("Plot saved as correlation_heatmap.png")
    else:
        print("Not enough numeric columns to generate a correlation heatmap.")

    # ------------------------------------------------
    # Step 10 : Data Preprocessing - Handle Zero Values
    # ------------------------------------------------
    print(border)
    print("Step 10 : Data Preprocessing - Handle Zero Values")
    print(border)

    zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    present_zero_columns = [col for col in zero_columns if col in df.columns]

    if present_zero_columns:
        print("Zero value counts before replacement :")
        for col in present_zero_columns:
            print(f"  {col} : {(df[col] == 0).sum()} zeros")

        # Replace 0 with column median
        for col in present_zero_columns:
            median_val = df[col].replace(0, np.nan).median()
            df[col] = df[col].replace(0, median_val)

        print("\nZero value counts after replacement (should all be 0) :")
        for col in present_zero_columns:
            print(f"  {col} : {(df[col] == 0).sum()} zeros")
    else:
        print("No biological zero-value columns found for replacement.")

    # ------------------------------------------------
    # Step 11 : Split Dataset into Features and Target
    # ------------------------------------------------
    print(border)
    print("Step 11 : Split Dataset into Features (X) and Target (y)")
    print(border)

    X = df.drop(columns=[target_column])
    X = X.select_dtypes(include='number')
    Y = df[target_column]

    print("Shape of Features (X) :", X.shape)
    print("Shape of Target   (y) :", Y.shape)

    if X.shape[1] == 0:
        print("No numeric feature columns available for modeling. Skipping training and prediction steps.")
        return

    if X.shape[1] == 0:
        print("No numeric feature columns available for modeling. Skipping training and prediction steps.")
        return

    # ------------------------------------------------
    # Step 12 : Apply Feature Scaling
    # ------------------------------------------------
    print(border)
    print("Step 12 : Apply Feature Scaling using StandardScaler")
    print(border)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Feature scaling applied successfully.")
    print("Sample scaled values (first row) :", X_scaled[0])

    # ------------------------------------------------
    # Step 13 : Split Dataset for Training and Testing
    # ------------------------------------------------
    print(border)
    print("Step 13 : Split Dataset for Training and Testing")
    print(border)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=0.2, random_state=42
    )

    print("X_train shape :", X_train.shape)
    print("X_test  shape :", X_test.shape)
    print("Y_train shape :", Y_train.shape)
    print("Y_test  shape :", Y_test.shape)

    # ------------------------------------------------
    # Step 14 : Build Model 1 - Logistic Regression
    # ------------------------------------------------
    print(border)
    print("Step 14 : Build Model 1 - Logistic Regression")
    print(border)

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, Y_train)
    lr_pred = lr_model.predict(X_test)
    print("Logistic Regression training complete.")

    # ------------------------------------------------
    # Step 15 : Build Model 2 - K-Nearest Neighbors
    # ------------------------------------------------
    print(border)
    print("Step 15 : Build Model 2 - K-Nearest Neighbors (KNN)")
    print(border)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, Y_train)
    knn_pred = knn_model.predict(X_test)
    print("KNN training complete.")

    # ------------------------------------------------
    # Step 16 : Build Model 3 - Decision Tree
    # ------------------------------------------------
    print(border)
    print("Step 16 : Build Model 3 - Decision Tree")
    print(border)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, Y_train)
    dt_pred = dt_model.predict(X_test)
    print("Decision Tree training complete.")

    # ------------------------------------------------
    # Step 17 : Evaluate All Models
    # ------------------------------------------------
    print(border)
    print("Step 17 : Evaluate All Models")
    print(border)

    models = {
        "Logistic Regression": lr_pred,
        "K-Nearest Neighbors": knn_pred,
        "Decision Tree"      : dt_pred
    }

    for model_name, y_pred in models.items():
        print(f"\n{'='*40}")
        print(f"  Model : {model_name}")
        print(f"{'='*40}")
        print(f"  Accuracy  : {accuracy_score(Y_test, y_pred):.4f}")
        print(f"  Precision : {precision_score(Y_test, y_pred):.4f}")
        print(f"  Recall    : {recall_score(Y_test, y_pred):.4f}")
        print(f"  F1 Score  : {f1_score(Y_test, y_pred):.4f}")
        print(f"\n  Classification Report :\n")
        print(classification_report(Y_test, y_pred,
                                    target_names=["Non-Diabetic", "Diabetic"]))

    # ------------------------------------------------
    # Step 18 : Visualize Confusion Matrices
    # ------------------------------------------------
    print(border)
    print("Step 18 : Visualize Confusion Matrices")
    print(border)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_names = list(models.keys())

    for ax, (model_name, y_pred) in zip(axes, models.items()):
        cm = confusion_matrix(Y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["Non-Diabetic", "Diabetic"],
                    yticklabels=["Non-Diabetic", "Diabetic"])
        ax.set_title(f"Confusion Matrix\n{model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    #plt.show()
    print("Plot saved as confusion_matrices.png")

    # ------------------------------------------------
    # Step 19 : Compare Model Accuracies (Bar Chart)
    # ------------------------------------------------
    print(border)
    print("Step 19 : Compare Model Accuracies")
    print(border)

    accuracy_scores = {
        name: accuracy_score(Y_test, pred)
        for name, pred in models.items()
    }

    print("Model Accuracy Summary :")
    for name, acc in accuracy_scores.items():
        print(f"  {name} : {acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.bar(accuracy_scores.keys(), accuracy_scores.values(),
            color=['steelblue', 'tomato', 'seagreen'])
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for i, (name, acc) in enumerate(accuracy_scores.items()):
        plt.text(i, acc + 0.01, f"{acc:.4f}", ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    #plt.show()
    print("Plot saved as model_comparison.png")

    # ------------------------------------------------
    # Step 20 : Final Output - Predict on Test Data
    # ------------------------------------------------
    print(border)
    print("Step 20 : Final Output - Predict on Test Data")
    print(border)

    # Use best model (Logistic Regression) for final predictions
    final_predictions = lr_model.predict(X_test)
    final_proba      = lr_model.predict_proba(X_test)[:, 1]

    Result = pd.DataFrame({
        'Actual Outcome'    : Y_test.values,
        'Predicted Outcome' : final_predictions,
        'Diabetic Probability' : np.round(final_proba, 4),
        'Result'            : ['Diabetic' if p == 1 else 'Non-Diabetic'
                               for p in final_predictions]
    })

    print("Sample Predictions (first 10 rows) :")
    print(Result.head(10))

    # Save predictions to CSV
    Result.to_csv("predictions.csv", index=False)
    print("\nAll predictions saved to predictions.csv")

    # ------------------------------------------------
    # Step 21 : Predict for a Single New Patient
    # ------------------------------------------------
    print(border)
    print("Step 21 : Predict for a Single New Patient")
    print(border)

    # Sample patient: [Pregnancies, Glucose, BloodPressure, SkinThickness,
    #                  Insulin, BMI, DiabetesPedigreeFunction, Age]
    new_patient = np.array([[ 0, 33.69]])
    new_patient_scaled = scaler.transform(new_patient)
    patient_pred       = lr_model.predict(new_patient_scaled)
    patient_proba      = lr_model.predict_proba(new_patient_scaled)[0][1]

    print("New Patient Data :")
    print("  Pregnancies             : 6")
    print("  Glucose                 : 148")
    print("  BloodPressure           : 72")
    print("  SkinThickness           : 35")
    print("  Insulin                 : 0")
    print("  BMI                     : 33.6")
    print("  DiabetesPedigreeFunction: 0.627")
    print("  Age                     : 50")
    print(f"\nPrediction       : {'Diabetic' if patient_pred[0] == 1 else 'Non-Diabetic'}")
    print(f"Probability      : {patient_proba:.4f}")


def main():
    DiabetesPrediction("diabetes.csv")


if __name__ == "__main__":
    main()