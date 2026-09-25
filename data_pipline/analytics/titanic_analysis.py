# ============================================================
# TITANIC ANALYSIS - ZEpto AI/ML CAPSTONE MODULE 2
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ============================================================
# 1. LOAD TITANIC DATASET
# ============================================================

print("\n============================================================")
print("LOADING TITANIC DATASET")
print("============================================================")

df = sns.load_dataset("titanic")

# Save immediately
df.to_csv("titanic.csv", index=False)

print("Dataset saved as titanic.csv")


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n============================================================")
print("BASIC DATASET INFORMATION")
print("============================================================")

print("\n--- INFO ---")
df.info()

print("\n--- DESCRIBE ---")
print(df.describe())

print("\n--- SHAPE ---")
print(df.shape)


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n============================================================")
print("MISSING VALUE ANALYSIS")
print("============================================================")

missing_report = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": df.isnull().mean() * 100
})

missing_report = missing_report[
    missing_report["Missing_Count"] > 0
]

print(missing_report)

missing_report.to_csv(
    os.path.join(OUTPUT_DIR, "missing_values.csv")
)


# ============================================================
# 4. EDA CLEANING
# ============================================================

print("\n============================================================")
print("EDA MISSING VALUE HANDLING")
print("============================================================")

eda_df = df.copy()

for column in eda_df.columns:

    missing_pct = eda_df[column].isnull().mean() * 100

    if missing_pct == 0:
        continue

    print(
        f"\n{column}: {missing_pct:.2f}% missing"
    )

    # Less than 5% -> drop rows
    if missing_pct < 5:

        eda_df = eda_df.dropna(subset=[column])

        print("Action: Dropped rows")

    # 5% to 30% -> impute
    elif missing_pct <= 30:

        if pd.api.types.is_numeric_dtype(
            eda_df[column]
        ):

            value = eda_df[column].median()

            eda_df[column] = eda_df[column].fillna(
                value
            )

            print(
                f"Action: Median imputation = {value:.2f}"
            )

        else:

            value = eda_df[column].mode()

            if len(value) > 0:
                fill_value = value.iloc[0]
            else:
                fill_value = "Missing"

            if str(eda_df[column].dtype) == "category":
                eda_df[column] = eda_df[column].astype(
                    "object"
                )

            eda_df[column] = eda_df[column].fillna(
                fill_value
            )

            print(
                f"Action: Mode imputation = {fill_value}"
            )

    # More than 30% -> encode as Missing
    else:

        if str(eda_df[column].dtype) == "category":
            eda_df[column] = eda_df[column].astype(
                "object"
            )

        eda_df[column] = eda_df[column].fillna(
            "Missing"
        )

        print(
            "Action: Encoded missing values as Missing"
        )


# ============================================================
# 5. AGE AND FARE DISTRIBUTION
# ============================================================

print("\n============================================================")
print("AGE AND FARE DISTRIBUTION")
print("============================================================")

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5)
)

axes[0].hist(
    eda_df["age"].dropna(),
    bins=20
)

axes[0].set_title("Age Distribution")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Frequency")

axes[1].hist(
    eda_df["fare"].dropna(),
    bins=20
)

axes[1].set_title("Fare Distribution")
axes[1].set_xlabel("Fare")
axes[1].set_ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "age_fare_distribution.png"
    )
)

plt.show()
plt.close()


# ============================================================
# 6. BOX PLOTS
# ============================================================

print("\nCreating boxplots...")

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5)
)

axes[0].boxplot(
    eda_df["age"].dropna()
)

axes[0].set_title("Age Boxplot")
axes[0].set_ylabel("Age")

axes[1].boxplot(
    eda_df["fare"].dropna()
)

axes[1].set_title("Fare Boxplot")
axes[1].set_ylabel("Fare")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "age_fare_boxplots.png"
    )
)

plt.show()
plt.close()


# ============================================================
# 7. IQR OUTLIERS
# ============================================================

print("\n============================================================")
print("IQR OUTLIER ANALYSIS")
print("============================================================")


def iqr_outliers(series):

    series = series.dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = (
        (series < lower) |
        (series > upper)
    ).sum()

    return count, lower, upper


age_outliers, age_lower, age_upper = iqr_outliers(
    eda_df["age"]
)

fare_outliers, fare_lower, fare_upper = iqr_outliers(
    eda_df["fare"]
)

print("Age outliers :", age_outliers)
print("Fare outliers:", fare_outliers)

print(
    f"Age IQR limits: {age_lower:.2f} to {age_upper:.2f}"
)

print(
    f"Fare IQR limits: {fare_lower:.2f} to {fare_upper:.2f}"
)


# ============================================================
# 8. FARE STATISTICS
# ============================================================

print("\n============================================================")
print("FARE STATISTICS")
print("============================================================")

fare_mean = eda_df["fare"].mean()
fare_median = eda_df["fare"].median()
fare_mode = eda_df["fare"].mode().iloc[0]
fare_skew = eda_df["fare"].skew()

print(f"Mean   : {fare_mean:.2f}")
print(f"Median : {fare_median:.2f}")
print(f"Mode   : {fare_mode:.2f}")
print(f"Skew   : {fare_skew:.4f}")

if fare_mean > fare_median:
    print("Interpretation: Fare is right-skewed.")
elif fare_mean < fare_median:
    print("Interpretation: Fare is left-skewed.")
else:
    print("Interpretation: Fare is approximately symmetric.")


# ============================================================
# 9. SURVIVAL ANALYSIS
# ============================================================

print("\n============================================================")
print("SURVIVAL RATE ANALYSIS")
print("============================================================")

male_rate = eda_df[
    eda_df["sex"] == "male"
]["survived"].mean()

female_rate = eda_df[
    eda_df["sex"] == "female"
]["survived"].mean()

print(
    f"Male survival rate   : {male_rate:.3f}"
)

print(
    f"Female survival rate : {female_rate:.3f}"
)


print("\nSurvival rate by passenger class:")

for pclass in sorted(
    eda_df["pclass"].dropna().unique()
):

    rate = eda_df[
        eda_df["pclass"] == pclass
    ]["survived"].mean()

    print(
        f"Class {int(pclass)}: {rate:.3f}"
    )


print("\nSurvival rate by sex and class:")

for sex in ["male", "female"]:

    for pclass in sorted(
        eda_df["pclass"].dropna().unique()
    ):

        subset = eda_df[
            (eda_df["sex"] == sex) &
            (eda_df["pclass"] == pclass)
        ]

        if len(subset) > 0:

            rate = subset["survived"].mean()

            print(
                f"{sex}, Class {int(pclass)}: {rate:.3f}"
            )


# Demonstrate OR condition
male_or_first = eda_df[
    (eda_df["sex"] == "male") |
    (eda_df["pclass"] == 1)
]

print(
    "\nRows satisfying male OR first-class:",
    len(male_or_first)
)


# ============================================================
# 10. CORRELATION ANALYSIS
# ============================================================

print("\n============================================================")
print("CORRELATION ANALYSIS")
print("============================================================")

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = eda_df[
    correlation_columns
].corr()

print("\nCorrelation matrix:")
print(corr_matrix)


# Heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_heatmap.png"
    )
)

plt.show()
plt.close()


# ------------------------------------------------------------
# FIX FOR READ-ONLY ARRAY ERROR
# ------------------------------------------------------------

corr_abs = corr_matrix.abs().copy()

# IMPORTANT:
# to_numpy().copy() creates a writable array
corr_values = corr_abs.to_numpy().copy()

# Remove diagonal
np.fill_diagonal(
    corr_values,
    0
)

corr_abs = pd.DataFrame(
    corr_values,
    index=corr_matrix.index,
    columns=corr_matrix.columns
)

# Find unique pairs
pairs = []

for i in range(len(corr_abs.columns)):

    for j in range(i + 1, len(corr_abs.columns)):

        pairs.append(
            (
                corr_abs.index[i],
                corr_abs.columns[j],
                corr_abs.iloc[i, j]
            )
        )

pairs.sort(
    key=lambda x: x[2],
    reverse=True
)

print("\nTop two absolute off-diagonal correlations:")

for item in pairs[:2]:

    print(
        f"{item[0]} vs {item[1]}: {item[2]:.4f}"
    )


# ============================================================
# 11. MULTIVARIATE CHART 1
# ============================================================

print("\nCreating multivariate chart 1...")

plt.figure(figsize=(8, 6))

sns.barplot(
    data=eda_df,
    x="pclass",
    y="survived",
    hue="sex"
)

plt.title(
    "Survival Rate by Class and Sex"
)

plt.ylabel("Survival Rate")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "multivariate_survival_class_sex.png"
    )
)

plt.show()
plt.close()

print(
    "Interpretation: Survival varies across passenger classes."
)
print(
    "Sex also shows differences in survival rates."
)
print(
    "The combination of class and sex provides more information."
)
print(
    "This demonstrates the usefulness of multivariate analysis."
)


# ============================================================
# 12. MULTIVARIATE CHART 2
# ============================================================

print("\nCreating multivariate chart 2...")

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=eda_df,
    x="pclass",
    y="fare"
)

plt.title(
    "Fare Distribution Across Passenger Classes"
)

plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "multivariate_fare_class.png"
    )
)

plt.show()
plt.close()

print(
    "Interpretation: Fare distributions differ by class."
)
print(
    "Higher classes generally have higher fares."
)
print(
    "Fare values also contain several extreme observations."
)
print(
    "Passenger class therefore helps explain fare variation."
)


# ============================================================
# 13. MULTIVARIATE CHART 3
# ============================================================

print("\nCreating multivariate chart 3...")

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=eda_df,
    x="age",
    y="fare",
    hue="survived",
    style="sex"
)

plt.title(
    "Age vs Fare by Survival and Sex"
)

plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "multivariate_age_fare_survival.png"
    )
)

plt.show()
plt.close()

print(
    "Interpretation: Age and fare do not form a strong simple linear pattern."
)
print(
    "Fare varies considerably among different age groups."
)
print(
    "Survival adds another dimension to the relationship."
)
print(
    "Sex provides additional information about the observations."
)


# ============================================================
# 14. MULTIVARIATE CHART 4
# ============================================================

print("\nCreating multivariate chart 4...")

plt.figure(figsize=(8, 6))

sns.countplot(
    data=eda_df,
    x="embarked",
    hue="survived"
)

plt.title(
    "Survival Count by Embarkation Port"
)

plt.xlabel("Embarkation Port")
plt.ylabel("Passenger Count")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "multivariate_embarked_survival.png"
    )
)

plt.show()
plt.close()

print(
    "Interpretation: Passenger counts differ by embarkation port."
)
print(
    "Survival counts also vary between the ports."
)
print(
    "This relationship should be considered with class and sex."
)
print(
    "Embarkation provides additional contextual information."
)


# ============================================================
# 15. EDA STANDARDIZATION
# ============================================================

print("\n============================================================")
print("EDA STANDARDIZATION")
print("============================================================")

standardized_df = eda_df.copy()

for column in ["age", "fare"]:

    mean_value = standardized_df[column].mean()
    std_value = standardized_df[column].std()

    standardized_df[
        column + "_z"
    ] = (
        standardized_df[column] - mean_value
    ) / std_value

    print(f"\n{column}:")

    print(
        f"Before -> Mean={mean_value:.4f}, "
        f"Std={std_value:.4f}"
    )

    print(
        f"After  -> Mean="
        f"{standardized_df[column + '_z'].mean():.4f}, "
        f"Std="
        f"{standardized_df[column + '_z'].std():.4f}"
    )


standardized_df[
    ["age_z", "fare_z"]
].to_csv(
    os.path.join(
        OUTPUT_DIR,
        "eda_standardized_age_fare.csv"
    ),
    index=False
)


# ============================================================
# 16. MACHINE LEARNING DATA
# ============================================================

print("\n============================================================")
print("MACHINE LEARNING")
print("============================================================")

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df["survived"]


# ============================================================
# 17. STRATIFIED TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows : {len(X_test)}"
)

print(
    "Stratified splitting preserves class proportions."
)


# ============================================================
# 18. PREPROCESSING
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_features
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 19. THREE CLASSIFIERS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            oob_score=True
        )
}


classification_results = []

trained_pipelines = {}


# ============================================================
# 20. TRAIN CLASSIFIERS
# ============================================================

for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    trained_pipelines[
        model_name
    ] = pipeline

    y_pred = pipeline.predict(
        X_test
    )

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    classification_results.append({

        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc

    })

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"AUC      : {auc:.4f}"
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )

    plt.show()
    plt.close()


# ============================================================
# 21. CLASSIFIER COMPARISON
# ============================================================

classification_df = pd.DataFrame(
    classification_results
)

print("\n============================================================")
print("CLASSIFIER COMPARISON")
print("============================================================")

print(
    classification_df.to_string(
        index=False
    )
)

classification_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "classifier_comparison.csv"
    ),
    index=False
)


# ============================================================
# 22. ROC CURVES
# ============================================================

print("\nCreating ROC curves...")

plt.figure(figsize=(8, 6))

for model_name, pipeline in trained_pipelines.items():

    probability = pipeline.predict_proba(
        X_test
    )[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} AUC={auc:.3f}"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curves"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curves.png"
    )
)

plt.show()
plt.close()


# ============================================================
# 23. DECISION TREE VISUALIZATION
# ============================================================

print("\nCreating Decision Tree visualization...")

tree_pipeline = trained_pipelines[
    "Decision Tree"
]

tree_model = tree_pipeline.named_steps[
    "classifier"
]

feature_names = (
    tree_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

plt.figure(figsize=(20, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=[
        "Not Survived",
        "Survived"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title(
    "Decision Tree Classifier"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "decision_tree.png"
    )
)

plt.show()
plt.close()


# ============================================================
# 24. CLASS IMBALANCE
# ============================================================

print("\n============================================================")
print("CLASS IMBALANCE")
print("============================================================")

print(
    "\nTraining class counts:"
)

print(
    y_train.value_counts()
)

print(
    "\nTraining class percentages:"
)

print(
    (y_train.value_counts(normalize=True) * 100)
)


# ============================================================
# 25. IMBALANCE - BASELINE
# ============================================================

baseline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

baseline.fit(
    X_train,
    y_train
)

baseline_pred = baseline.predict(
    X_test
)


# ============================================================
# 26. IMBALANCE - CLASS WEIGHT
# ============================================================

balanced = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

balanced.fit(
    X_train,
    y_train
)

balanced_pred = balanced.predict(
    X_test
)


# ============================================================
# 27. IMBALANCE - SMOTE
# ============================================================

print("\nRunning SMOTE...")

smote_pipeline = ImbPipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "smote",
            SMOTE(
                random_state=42
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

smote_pipeline.fit(
    X_train,
    y_train
)

smote_pred = smote_pipeline.predict(
    X_test
)


# ============================================================
# 28. IMBALANCE COMPARISON
# ============================================================

imbalance_results = [

    {
        "Method": "Baseline",
        "Precision": precision_score(
            y_test,
            baseline_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            baseline_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            baseline_pred,
            zero_division=0
        )
    },

    {
        "Method": "Class Weight Balanced",
        "Precision": precision_score(
            y_test,
            balanced_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            balanced_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            balanced_pred,
            zero_division=0
        )
    },

    {
        "Method": "SMOTE",
        "Precision": precision_score(
            y_test,
            smote_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            smote_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            smote_pred,
            zero_division=0
        )
    }
]

imbalance_df = pd.DataFrame(
    imbalance_results
)

print(
    imbalance_df.to_string(
        index=False
    )
)

imbalance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "imbalance_comparison.csv"
    ),
    index=False
)


# ============================================================
# 29. RANDOM FOREST GRID SEARCH
# ============================================================

print("\n============================================================")
print("RANDOM FOREST GRID SEARCH")
print("============================================================")

rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                oob_score=True
            )
        )
    ]
)

param_grid = {

    "classifier__n_estimators": [
        100,
        200
    ],

    "classifier__max_depth": [
        None,
        5,
        10
    ],

    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

best_rf_pipeline = (
    grid_search.best_estimator_
)

print(
    "\nBest parameters:"
)

print(
    grid_search.best_params_
)

oob_score = (
    best_rf_pipeline
    .named_steps["classifier"]
    .oob_score_
)

print(
    f"OOB Score: {oob_score:.4f}"
)


# ============================================================
# 30. FARE REGRESSION
# ============================================================

print("\n============================================================")
print("FARE REGRESSION")
print("============================================================")

regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "survived",
    "embarked"
]

regression_df = df[
    regression_features + ["fare"]
].copy()

regression_df = regression_df.dropna(
    subset=["fare"]
)

X_reg = regression_df[
    regression_features
]

y_reg = regression_df[
    "fare"
]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)


reg_numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "survived"
]

reg_categorical_features = [
    "sex",
    "embarked"
]

reg_numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

reg_categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

reg_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            reg_numeric_transformer,
            reg_numeric_features
        ),
        (
            "categorical",
            reg_categorical_transformer,
            reg_categorical_features
        )
    ]
)

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            reg_preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

y_reg_pred = regression_pipeline.predict(
    X_reg_test
)


# ============================================================
# 31. REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

# Number of predictors after preprocessing
transformed_test = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_reg_test)
)

n = len(y_reg_test)
p = transformed_test.shape[1]

if n > p + 1:

    adjusted_r2 = (
        1 -
        (
            (1 - r2) *
            (n - 1) /
            (n - p - 1)
        )
    )

else:

    adjusted_r2 = np.nan


print(
    f"\nMAE         : {mae:.4f}"
)

print(
    f"RMSE        : {rmse:.4f}"
)

print(
    f"R2          : {r2:.4f}"
)

print(
    f"Adjusted R2 : {adjusted_r2:.4f}"
)


# ============================================================
# 32. RESIDUAL PLOT
# ============================================================

residuals = (
    y_reg_test - y_reg_pred
)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")

plt.title(
    "Residual Plot - Fare Regression"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "fare_regression_residuals.png"
    )
)

plt.show()
plt.close()


# ============================================================
# 33. HETEROSCEDASTICITY
# ============================================================

print("\nHeteroscedasticity analysis:")

predicted_series = pd.Series(
    y_reg_pred,
    index=residuals.index
)

residual_abs = residuals.abs()

residual_correlation = residual_abs.corr(
    predicted_series
)

if residual_correlation > 0.30:

    hetero_statement = (
        "The residual spread increases with "
        "predicted fare, suggesting possible "
        "heteroscedasticity."
    )

else:

    hetero_statement = (
        "The residual plot does not show a "
        "strong clear pattern of increasing variance."
    )

print(
    hetero_statement
)


# ============================================================
# 34. REGRESSION RESULTS
# ============================================================

regression_results = pd.DataFrame({

    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2],
    "Adjusted_R2": [adjusted_r2]

})

regression_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "regression_metrics.csv"
    ),
    index=False
)


# ============================================================
# 35. FINAL MODEL COMPARISON
# ============================================================

final_classifier = classification_df.copy()

final_classifier[
    "Regression_MAE"
] = np.nan

final_classifier[
    "Regression_RMSE"
] = np.nan

final_classifier[
    "Regression_R2"
] = np.nan

final_classifier[
    "Regression_Adjusted_R2"
] = np.nan


regression_row = pd.DataFrame({

    "Model": ["Linear Regression"],

    "Accuracy": [np.nan],

    "Precision": [np.nan],

    "Recall": [np.nan],

    "F1": [np.nan],

    "AUC": [np.nan],

    "Regression_MAE": [mae],

    "Regression_RMSE": [rmse],

    "Regression_R2": [r2],

    "Regression_Adjusted_R2": [
        adjusted_r2
    ]

})


final_comparison = pd.concat(
    [
        final_classifier,
        regression_row
    ],
    ignore_index=True
)

print("\n============================================================")
print("FINAL MODEL COMPARISON")
print("============================================================")

print(
    final_comparison.to_string(
        index=False
    )
)

final_comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 36. SELECT BEST CLASSIFIER
# ============================================================

best_row = classification_df.loc[
    classification_df["F1"].idxmax()
]

best_model_name = best_row["Model"]

print("\n============================================================")
print("BEST CLASSIFIER")
print("============================================================")

print(
    f"Selected model: {best_model_name}"
)

print(
    f"Accuracy : {best_row['Accuracy']:.4f}"
)

print(
    f"Precision: {best_row['Precision']:.4f}"
)

print(
    f"Recall   : {best_row['Recall']:.4f}"
)

print(
    f"F1       : {best_row['F1']:.4f}"
)

print(
    f"AUC      : {best_row['AUC']:.4f}"
)


# ============================================================
# 37. SAVE BEST COMPLETE PIPELINE
# ============================================================

if best_model_name == "Random Forest":

    final_pipeline = best_rf_pipeline

else:

    final_pipeline = trained_pipelines[
        best_model_name
    ]


# Retrain complete pipeline
final_pipeline.fit(
    X_train,
    y_train
)

pipeline_path = os.path.join(
    OUTPUT_DIR,
    "best_titanic_pipeline.joblib"
)

joblib.dump(
    final_pipeline,
    pipeline_path
)

print(
    f"\nPipeline saved to:\n{pipeline_path}"
)


# ============================================================
# 38. RELOAD AND PREDICT RAW INPUT
# ============================================================

print("\n============================================================")
print("RELOADING SAVED PIPELINE")
print("============================================================")

loaded_pipeline = joblib.load(
    pipeline_path
)

raw_input = X_test.iloc[[0]]

prediction = loaded_pipeline.predict(
    raw_input
)

probability = loaded_pipeline.predict_proba(
    raw_input
)[:, 1]

print("\nRaw input:")
print(
    raw_input.to_string(
        index=False
    )
)

print(
    "\nPredicted survival:",
    int(prediction[0])
)

print(
    f"Survival probability: "
    f"{probability[0]:.4f}"
)


# ============================================================
# 39. FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("FINAL MODEL SUMMARY")
print("============================================================")

print(
    f"Best classifier: {best_model_name}"
)

print(
    f"F1 Score: {best_row['F1']:.4f}"
)

print(
    f"Accuracy: {best_row['Accuracy']:.4f}"
)

print(
    f"Precision: {best_row['Precision']:.4f}"
)

print(
    f"Recall: {best_row['Recall']:.4f}"
)

print(
    f"AUC: {best_row['AUC']:.4f}"
)

print(
    f"\nFare regression MAE: {mae:.4f}"
)

print(
    f"Fare regression RMSE: {rmse:.4f}"
)

print(
    f"Fare regression R2: {r2:.4f}"
)

print(
    f"Fare regression Adjusted R2: "
    f"{adjusted_r2:.4f}"
)

print(
    "\nClassifier and regression metrics "
    "represent different prediction tasks."
)


# ============================================================
# COMPLETION
# ============================================================

print("\n============================================================")
print("TITANIC ANALYSIS COMPLETED SUCCESSFULLY")
print("============================================================")

print(
    f"\nAll files are saved inside: {OUTPUT_DIR}"
)