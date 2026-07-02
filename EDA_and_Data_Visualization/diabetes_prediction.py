import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

border = "-"*40


def load_and_explore(path):
    print(border)
    print("Step 1 : Load and explore dataset")
    print(border)

    df = pd.read_csv(path)

    print("First 5 rows :")
    print(df.head())

    print("Column info :")
    print(df.info())

    print("Null values :")
    print(df.isnull().sum())

    print("Basic statistics :")
    print(df.describe())

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Outcome", data=df)
    plt.title("Class Distribution")
    plt.savefig("outcome_distribution.png")
    plt.close()

    df.hist(figsize=(12, 10))
    plt.savefig("feature_histograms.png")
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df.drop("Outcome", axis=1))
    plt.xticks(rotation=45)
    plt.title("Boxplot of Features")
    plt.tight_layout()
    plt.savefig("feature_boxplot.png")
    plt.close()

    return df


def preprocess(df):
    print(border)
    print("Step 2 : Preprocess data")
    print(border)

    zero_not_valid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in zero_not_valid:
        df[col] = df[col].replace(0, df[col].mean())

    print("Replaced zero values in", zero_not_valid, "with column mean")

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Feature scaling done using StandardScaler")

    return X_scaled, y, X.columns


def train_and_evaluate(X_train, X_test, y_train, y_test, name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(border)
    print(name)
    print(border)
    print("Accuracy  :", acc)
    print("Precision :", prec)
    print("Recall    :", rec)
    print("F1 Score  :", f1)
    print("Confusion Matrix :")
    print(cm)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(name.replace(" ", "_") + "_confusion_matrix.png")
    plt.close()

    return model, acc


def main():
    df = load_and_explore("diabetes.csv")
    X, y, columns = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(border)
    print("Step 3 : Train models")
    print(border)

    log_model, log_acc = train_and_evaluate(X_train, X_test, y_train, y_test, "Logistic Regression", LogisticRegression(max_iter=1000))
    knn_model, knn_acc = train_and_evaluate(X_train, X_test, y_train, y_test, "KNN", KNeighborsClassifier(n_neighbors=5))
    tree_model, tree_acc = train_and_evaluate(X_train, X_test, y_train, y_test, "Decision Tree", DecisionTreeClassifier(random_state=42))

    print(border)
    print("Step 4 : Final predictions")
    print(border)

    best_model = log_model
    best_name = "Logistic Regression"
    best_acc = log_acc

    if knn_acc > best_acc:
        best_model, best_name, best_acc = knn_model, "KNN", knn_acc
    if tree_acc > best_acc:
        best_model, best_name, best_acc = tree_model, "Decision Tree", tree_acc

    print("Best model :", best_name, "with accuracy", best_acc)

    final_pred = best_model.predict(X_test)

    results = pd.DataFrame(X_test, columns=columns)
    results["Actual"] = y_test.values
    results["Predicted"] = final_pred
    results["Predicted"] = results["Predicted"].map({0: "Not Diabetic", 1: "Diabetic"})

    print(results.head(10))

    results.to_csv("diabetes_predictions.csv", index=False)
    print("Predictions saved to diabetes_predictions.csv")


if __name__ == "__main__":
    main()