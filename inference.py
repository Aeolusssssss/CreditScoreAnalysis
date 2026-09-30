"""
inference.py
------------
Load model + preprocessor yang sudah disimpan pipeline.py, lalu prediksi
credit score dari satu nasabah.

Dipakai oleh app Streamlit (app.py) dan bisa juga dijalankan langsung
buat ngetes:  python inference.py
"""

import pickle
from pathlib import Path

import pandas as pd

# class DataPreprocessor dipanggil saat unpickle preprocessor,
# jadi harus ke-import dari sini.
from pipeline import DataPreprocessor  # noqa: F401

MODEL_DIR = Path("model")


class CreditScorePredictor:
    def __init__(self, model_dir=MODEL_DIR):
        with open(Path(model_dir) / "best_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(Path(model_dir) / "preprocessor.pkl", "rb") as f:
            self.pre = pickle.load(f)
        self.classes = list(self.pre.label_encoder.classes_)

    def predict(self, data: dict):
        """
        data : dict berisi fitur satu nasabah.
        return: (label, dict probabilitas per kelas)
        """
        df = pd.DataFrame([data])
        X = self.pre.transform(df)
        pred = self.model.predict(X)[0]
        label = self.pre.decode_target([pred])[0]

        proba = {}
        if hasattr(self.model, "predict_proba"):
            p = self.model.predict_proba(X)[0]
            proba = {self.classes[i]: float(p[i]) for i in range(len(self.classes))}
        return label, proba


# Test case: tiap dict mewakili satu kelas (diambil dari data asli).
TEST_CASES = {
    "Good": {
        "Age": 29, "Occupation": "Doctor", "Annual_Income": 18002.67,
        "Monthly_Inhand_Salary": 1430.22, "Num_Bank_Accounts": 6,
        "Num_Credit_Card": 3, "Interest_Rate": 10, "Num_of_Loan": 3,
        "Delay_from_due_date": 27, "Num_of_Delayed_Payment": 15,
        "Changed_Credit_Limit": 0.94, "Num_Credit_Inquiries": 6,
        "Credit_Mix": "Good", "Outstanding_Debt": 698.05,
        "Credit_Utilization_Ratio": 29.98, "Credit_History_Months": 339,
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 34.03,
        "Amount_invested_monthly": 32.73,
        "Payment_Behaviour": "High_spent_Large_value_payments",
        "Monthly_Balance": 316.26, "Num_Loan_Types": 3,
    },
    "Standard": {
        "Age": 29, "Occupation": "Musician", "Annual_Income": 142081.48,
        "Monthly_Inhand_Salary": 11771.12, "Num_Bank_Accounts": 6,
        "Num_Credit_Card": 6, "Interest_Rate": 15, "Num_of_Loan": 3,
        "Delay_from_due_date": 29, "Num_of_Delayed_Payment": 20,
        "Changed_Credit_Limit": 8.21, "Num_Credit_Inquiries": 4,
        "Credit_Mix": "Standard", "Outstanding_Debt": 932.32,
        "Credit_Utilization_Ratio": 28.41, "Credit_History_Months": 324,
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 180.62,
        "Amount_invested_monthly": 423.96,
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": 862.53, "Num_Loan_Types": 3,
    },
    "Poor": {
        "Age": 46, "Occupation": "Writer", "Annual_Income": 83823.22,
        "Monthly_Inhand_Salary": 6841.27, "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 5, "Interest_Rate": 11, "Num_of_Loan": 1,
        "Delay_from_due_date": 6, "Num_of_Delayed_Payment": 9,
        "Changed_Credit_Limit": 10.31, "Num_Credit_Inquiries": 0,
        "Credit_Mix": "Good", "Outstanding_Debt": 184.71,
        "Credit_Utilization_Ratio": 37.6, "Credit_History_Months": 245,
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 41.3,
        "Amount_invested_monthly": 451.55,
        "Payment_Behaviour": "Low_spent_Medium_value_payments",
        "Monthly_Balance": 471.27, "Num_Loan_Types": 1,
    },
}


if __name__ == "__main__":
    predictor = CreditScorePredictor()
    print("=== Test inferensi per kelas ===\n")
    for expected, case in TEST_CASES.items():
        label, proba = predictor.predict(case)
        conf = proba.get(label, 0)
        status = "OK" if label == expected else "MISS"
        print(f"[{status}] expected={expected:9s} -> pred={label:9s} "
              f"(conf={conf:.2%})")
