import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# 1. Load Dataset
# -----------------------------------

data_path = "dataset/train_u6lujuX_CVtuZ9i (1).csv"

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# -----------------------------------
# 2. Remove Loan ID
# -----------------------------------

df = df.drop("Loan_ID", axis=1)


# -----------------------------------
# 3. Handle Missing Values
# -----------------------------------

categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed"
]

numerical_columns = [
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]

# Fill categorical missing values
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Fill numerical missing values
for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())


# -----------------------------------
# 4. Encode Categorical Data
# -----------------------------------

label_encoder = LabelEncoder()

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

df["Education"] = label_encoder.fit_transform(df["Education"])
df["Property_Area"] = label_encoder.fit_transform(df["Property_Area"])

# Convert Loan Status
df["Loan_Status"] = df["Loan_Status"].map({
    "Y": 1,
    "N": 0
})


# -----------------------------------
# 5. Separate Features and Target
# -----------------------------------

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]


# -----------------------------------
# 6. Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

from imblearn.over_sampling import SMOTE

print("Original Training Data Shape:", X_train.shape)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Balanced Training Data Shape (After SMOTE):", X_train_smote.shape)
# -----------------------------------------

# -----------------------------------
# 7. Train Random Forest Model
# -----------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

# -----------------------------------
# 8. Evaluate Model
# -----------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------
# 9. Save Model
# -----------------------------------

model_path = "model/loan_model.pkl"

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Location:", model_path)