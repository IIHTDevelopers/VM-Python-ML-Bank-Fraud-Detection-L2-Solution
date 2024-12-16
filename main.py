import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE


def load_and_preprocess_data(dataset_path):
    df = pd.read_csv(dataset_path)

    # Encode categorical features
    encoder = OneHotEncoder(drop='first')
    categorical_features = ["TransactionType", "Location", "Merchant"]
    encoded_features = encoder.fit_transform(df[categorical_features]).toarray()

    # Combine with numerical features
    numerical_features = df[["TransactionAmount"]]
    X = np.hstack((numerical_features, encoded_features))

    # Target variable
    y = df["IsFraud"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return df, X_train, X_test, y_train, y_test


def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


def train_xgboost_model(X_train, y_train):
    xgb_model = xgb.XGBClassifier(scale_pos_weight=20, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    return xgb_model


def evaluate_model(xgb_model, X_test, y_test, threshold=0.2):
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nROC-AUC Score:")
    print(roc_auc_score(y_test, y_prob))


def save_model(xgb_model, model_path='xgboost_model.pkl'):
    joblib.dump(xgb_model, model_path)


# Analytical questions as individual functions
def number_of_fraudulent_accounts(df):
    return df[df['IsFraud'] == 1]['AccountID'].nunique()


def total_fraudulent_transactions(df):
    return df['IsFraud'].sum()


def high_risk_location(df):
    return df[df['IsFraud'] == 1]['Location'].mode()[0]


def high_risk_transaction_type(df):
    return df[df['IsFraud'] == 1]['TransactionType'].mode()[0]


def merchant_with_highest_fraud(df):
    return df[df['IsFraud'] == 1]['Merchant'].mode()[0]


def average_fraud_transaction_amount(df):
    return df[df['IsFraud'] == 1]['TransactionAmount'].mean()


def total_fraud_transaction_amount(df):
    return df[df['IsFraud'] == 1]['TransactionAmount'].sum()


def number_of_unique_fraudulent_transactions(df):
    return df[df['IsFraud'] == 1]['TransactionID'].nunique()


# Example usage
dataset_path = "credit_card_fraud_dataset.csv"
df, X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
X_train_resampled, y_train_resampled = balance_data(X_train, y_train)
xgb_model = train_xgboost_model(X_train_resampled, y_train_resampled)
save_model(xgb_model)
evaluate_model(xgb_model, X_test, y_test)

# Analytical Results
print("\nAnalytical Results:")
print(f"Number of Fraudulent Accounts: {number_of_fraudulent_accounts(df)}")
print(f"Total Fraudulent Transactions: {total_fraudulent_transactions(df)}")
print(f"High-Risk Location: {high_risk_location(df)}")
print(f"High-Risk Transaction Type: {high_risk_transaction_type(df)}")
print(f"Merchant with Highest Fraud: {merchant_with_highest_fraud(df)}")
print(f"Average Fraud Transaction Amount: {average_fraud_transaction_amount(df):.2f}")
print(f"Total Fraud Transaction Amount: {total_fraud_transaction_amount(df):.2f}")
print(f"Number of Unique Fraudulent Transactions: {number_of_unique_fraudulent_transactions(df)}")
