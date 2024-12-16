import unittest
import pandas as pd
from templatebank import (
    number_of_fraudulent_accounts, total_fraudulent_transactions,
    high_risk_location, high_risk_transaction_type,
    merchant_with_highest_fraud, average_fraud_transaction_amount,
    total_fraud_transaction_amount, number_of_unique_fraudulent_transactions
)
from test.TestUtils import TestUtils


class TestFraudAnalysis(unittest.TestCase):
    def setUp(self):
        # Replace 'Credit_Card_Fraud_Dataset.csv' with the actual path to your CSV file
        csv_file_path = 'Credit_Card_Fraud_Dataset.csv'

        # Read the CSV into a pandas DataFrame
        try:
            self.df = pd.read_csv(csv_file_path)
        except FileNotFoundError as e:
            print(f"Error: CSV file not found at {csv_file_path}. Please check the file path.")
            raise e
        except pd.errors.EmptyDataError as e:
            print(f"Error: CSV file at {csv_file_path} is empty. Please provide valid data.")
            raise e
        except Exception as e:
            print(f"An error occurred while reading the CSV file: {str(e)}")
            raise e

        self.expected_fraudulent_accounts = 16
        self.expected_total_fraudulent_transactions = 21
        self.expected_high_risk_location = 'California'
        self.expected_high_risk_transaction_type = 'Online'
        self.expected_merchant_with_highest_fraud = 'Target'
        self.expected_average_fraud_transaction_amount = 2747.48
        self.expected_total_fraud_transaction_amount = 57696.99
        self.expected_unique_fraudulent_transactions = 21

    def test_number_of_fraudulent_accounts(self):
        test_obj = TestUtils()
        try:
            result = number_of_fraudulent_accounts(self.df)
            print(f"TestNumberOfFraudulentAccounts - Expected: {self.expected_fraudulent_accounts}, Actual: {result}")
            if result == self.expected_fraudulent_accounts:
                test_obj.yakshaAssert("TestNumberOfFraudulentAccounts", True, "functional")
                print("TestNumberOfFraudulentAccounts: Passed")
            else:
                test_obj.yakshaAssert("TestNumberOfFraudulentAccounts", False, "functional")
                print("TestNumberOfFraudulentAccounts: Failed")
        except Exception as e:
            test_obj.yakshaAssert("TestNumberOfFraudulentAccounts", False, "functional")
            print(f"TestNumberOfFraudulentAccounts: Failed with exception {str(e)}")

    def test_total_fraudulent_transactions(self):
        test_obj = TestUtils()
        try:
            result = total_fraudulent_transactions(self.df)
            print(f"TestTotalFraudulentTransactions - Expected: {self.expected_total_fraudulent_transactions}, Actual: {result}")
            if result == self.expected_total_fraudulent_transactions:
                test_obj.yakshaAssert("TestTotalFraudulentTransactions", True, "functional")
                print("TestTotalFraudulentTransactions: Passed")
            else:
                test_obj.yakshaAssert("TestTotalFraudulentTransactions", False, "functional")
                print("TestTotalFraudulentTransactions: Failed")
        except Exception as e:
            test_obj.yakshaAssert("TestTotalFraudulentTransactions", False, "functional")
            print(f"TestTotalFraudulentTransactions: Failed with exception {str(e)}")

    def test_high_risk_location(self):
        test_obj = TestUtils()
        try:
            result = high_risk_location(self.df)
            print(f"TestHighRiskLocation - Expected: {self.expected_high_risk_location}, Actual: {result}")
            if result == self.expected_high_risk_location:
                test_obj.yakshaAssert("TestHighRiskLocation", True, "functional")
                print("TestHighRiskLocation: Passed")
            else:
                test_obj.yakshaAssert("TestHighRiskLocation", False, "functional")
                print("TestHighRiskLocation: Failed")
        except Exception as e:
            test_obj.yakshaAssert("TestHighRiskLocation", False, "functional")
            print(f"TestHighRiskLocation: Failed with exception {str(e)}")

    def test_high_risk_transaction_type(self):
        test_obj = TestUtils()
        try:
            result = high_risk_transaction_type(self.df)
            print(f"TestHighRiskTransactionType - Expected: {self.expected_high_risk_transaction_type}, Actual: {result}")
            if result == self.expected_high_risk_transaction_type:
                test_obj.yakshaAssert("TestHighRiskTransactionType", True, "functional")
                print("TestHighRiskTransactionType: Passed")
            else:
                test_obj.yakshaAssert("TestHighRiskTransactionType", False, "functional")
                print("TestHighRiskTransactionType: Failed")
        except Exception as e:
            test_obj.yakshaAssert("TestHighRiskTransactionType", False, "functional")
            print(f"TestHighRiskTransactionType: Failed with exception {str(e)}")

    def test_merchant_with_highest_fraud(self):
        test_obj = TestUtils()
        try:
            result = merchant_with_highest_fraud(self.df)
            print(f"TestMerchantWithHighestFraud - Expected: {self.expected_merchant_with_highest_fraud}, Actual: {result}")
            if result == self.expected_merchant_with_highest_fraud:
                test_obj.yakshaAssert("TestMerchantWithHighestFraud", True, "functional")
                print("TestMerchantWithHighestFraud: Passed")
            else:
                test_obj.yakshaAssert("TestMerchantWithHighestFraud", False, "functional")
                print("TestMerchantWithHighestFraud: Failed")
        except Exception as e:
            test_obj.yakshaAssert("TestMerchantWithHighestFraud", False, "functional")
            print(f"TestMerchantWithHighestFraud: Failed with exception {str(e)}")

    def test_average_fraud_transaction_amount(self):
        test_obj = TestUtils()
        try:
            result = average_fraud_transaction_amount(self.df)
            print(f"TestAverageFraudTransactionAmount - Expected: {self.expected_average_fraud_transaction_amount}, Actual: {result}")
            if round(result, 2) == round(self.expected_average_fraud_transaction_amount, 2):
                test_obj.yakshaAssert("TestAverageFraudTransactionAmount", True, "functional")
                print("TestAverageFraudTransactionAmount: Passed")
            else:
                test_obj.yakshaAssert("TestAverageFraudTransactionAmount", False, "functional")
                print("TestAverageFraudTransactionAmount: Failed")
        except Exception as e:
            test_obj.yakshaAssert("TestAverageFraudTransactionAmount", False, "functional")
            print(f"TestAverageFraudTransactionAmount: Failed with exception {str(e)}")

    def test_total_fraud_transaction_amount(self):
        test_obj = TestUtils()
        try:
            result = total_fraud_transaction_amount(self.df)
            print(f"TestTotalFraudTransactionAmount - Expected: {self.expected_total_fraud_transaction_amount}, Actual: {result}")
            if round(result, 2) == round(self.expected_total_fraud_transaction_amount, 2):
                test_obj.yakshaAssert("TestTotalFraudTransactionAmount", True, "functional")
                print("TestTotalFraudTransactionAmount: Passed")
            else:
                test_obj.yakshaAssert("TestTotalFraudTransactionAmount", False, "functional")
                print("TestTotalFraudTransactionAmount: Failed")
        except Exception as e:
            test_obj.yakshaAssert("TestTotalFraudTransactionAmount", False, "functional")
            print(f"TestTotalFraudTransactionAmount: Failed with exception {str(e)}")

    def test_number_of_unique_fraudulent_transactions(self):
        test_obj = TestUtils()
        try:
            result = number_of_unique_fraudulent_transactions(self.df)
            print(f"TestNumberOfUniqueFraudulentTransactions - Expected: {self.expected_unique_fraudulent_transactions}, Actual: {result}")
            if result == self.expected_unique_fraudulent_transactions:
                test_obj.yakshaAssert("TestNumberOfUniqueFraudulentTransactions", True, "functional")
                print("TestNumberOfUniqueFraudulentTransactions: Passed")
            else:
                test_obj.yakshaAssert("TestNumberOfUniqueFraudulentTransactions", False, "functional")
                print("TestNumberOfUniqueFraudulentTransactions: Failed")
        except Exception as e:
            test_obj.yakshaAssert("TestNumberOfUniqueFraudulentTransactions", False, "functional")
            print(f"TestNumberOfUniqueFraudulentTransactions: Failed with exception {str(e)}")


if __name__ == '__main__':
    unittest.main()
