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

def number_of_fraudulent_accounts(df):
    # Return the number of unique accounts involved in fraudulent transactions
    pass

def total_fraudulent_transactions(df):
    # Return the total count of fraudulent transactions
    pass

def high_risk_location(df):
    # Return the location with the highest number of fraudulent transactions
    pass

def high_risk_transaction_type(df):
    # Return the transaction type with the highest number of fraudulent occurrences
    pass

def merchant_with_highest_fraud(df):
    # Return the merchant associated with the highest number of fraudulent transactions
    pass

def average_fraud_transaction_amount(df):
    # Return the average transaction amount for fraudulent transactions
    pass

def total_fraud_transaction_amount(df):
    # Return the total transaction amount for fraudulent transactions
    pass

def number_of_unique_fraudulent_transactions(df):
    # Return the number of unique fraudulent transactions
    pass

# Example usage for later implementation
if __name__ == "__main__":
    dataset_path = "Credit_Card_Fraud_Dataset.csv"  # Replace with your actual dataset path
    df = pd.read_csv(dataset_path)

    # Call the functions and print results (implementations need to be added)
    print("Number of Fraudulent Accounts:", number_of_fraudulent_accounts(df))
    print("Total Fraudulent Transactions:", total_fraudulent_transactions(df))
    print("High-Risk Location:", high_risk_location(df))
    print("High-Risk Transaction Type:", high_risk_transaction_type(df))
    print("Merchant with Highest Fraud:", merchant_with_highest_fraud(df))
    print("Average Fraud Transaction Amount:", average_fraud_transaction_amount(df))
    print("Total Fraud Transaction Amount:", total_fraud_transaction_amount(df))
    print("Number of Unique Fraudulent Transactions:", number_of_unique_fraudulent_transactions(df))
