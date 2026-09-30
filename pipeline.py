"""
pipeline.py
-----------
Pipeline training model credit scoring (dataset A).

Strukturnya OOP supaya gampang di-retrain dan tiap tahap kepisah jelas:
  - DataPreprocessor : cleaning + transform fitur (fit/transform ala sklearn)
  - ModelTrainer     : training beberapa kandidat model
  - ModelEvaluator   : hitung metrik evaluasi
  - CreditScorePipeline : orkestrasi semuanya + logging ke MLflow

Jalankan: python pipeline.py
Output   : model/best_model.pkl, model/preprocessor.pkl, model/metadata.json
           + experiment tercatat di folder ./mlruns (buka pakai `mlflow ui`)
"""

import re
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = "data_A.csv"
MODEL_DIR = Path("model")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. PREPROCESSING
# ---------------------------------------------------------------------------
class DataPreprocessor:
    """
    Bertanggung jawab membersihkan data mentah dan mengubahnya jadi matriks
    fitur numerik yang siap dipakai model.

    Data dari institusi ini banyak noise-nya: angka kebawa underscore
    (mis. '39_'), placeholder sampah ('_______', '_', '!@9#%8'), dan outlier
    yang jelas salah input (umur 8000an, jumlah kartu kredit ratusan). Semua
    itu dihandle di sini, lalu di-impute dan di-encode.

    Dipakai dua kali: saat training (fit) dan saat inferensi (transform),
    jadi statistik imputasi & encoder konsisten.
    """

    # kolom identitas / tidak dipakai sebagai fitur
    DROP_COLS = ["ID", "Customer_ID", "Name", "SSN", "Month"]

    # kolom yang seharusnya numerik tapi kebaca string gara-gara underscore dsb
    DIRTY_NUMERIC = [
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt",
        "Amount_invested_monthly", "Monthly_Balance",
    ]

    # token yang dianggap missing
    GARBAGE = {"_______", "_", "!@9#%8", "#F%$D@*&8", "nan", "NaN", ""}

    # batas wajar -> di luar ini dianggap salah input, jadiin NaN lalu diimpute
    OUTLIER_BOUNDS = {
        "Age": (14, 100),
        "Num_Bank_Accounts": (0, 15),
        "Num_Credit_Card": (0, 15),
        "Interest_Rate": (1, 40),
        "Num_of_Loan": (0, 12),
        "Num_of_Delayed_Payment": (0, 50),
        "Num_Credit_Inquiries": (0, 50),
        "Delay_from_due_date": (-10, 120),
    }

    CATEGORICAL = ["Occupation", "Credit_Mix", "Payment_of_Min_Amount",
                   "Payment_Behaviour"]

    def __init__(self):
        self.column_transformer = None
        self.numeric_cols = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.fitted = False

    # ---- helper cleaning (dipakai fit & transform) ----
    @staticmethod
    def _credit_history_to_months(val):
        if pd.isna(val):
            return np.nan
        m = re.search(r"(\d+)\s*Year.*?(\d+)\s*Month", str(val))
        if not m:
            return np.nan
        return int(m.group(1)) * 12 + int(m.group(2))

    @staticmethod
    def _count_loan_types(val):
        # Type_of_Loan multi-value, dipisah koma & 'and'
        if pd.isna(val):
            return 0
        parts = re.split(r",\s*and\s*|,\s*|\s+and\s+", str(val))
        parts = [p.strip() for p in parts if p.strip() and p.strip().lower() != "and"]
        return len(parts)

    def _clean(self, df):
        """Cleaning umum, balikin dataframe dengan kolom fitur final."""
        df = df.copy()

        # buang spasi & token sampah jadi NaN dulu (kolom object)
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace(list(self.GARBAGE), np.nan)

        # numeric kotor -> strip underscore, coerce
        for c in self.DIRTY_NUMERIC:
            if c in df.columns:
                df[c] = pd.to_numeric(
                    df[c].astype(str).str.replace("_", "", regex=False),
                    errors="coerce",
                )

        # fitur turunan. Dibuat fleksibel: kalau kolom mentah ada -> diturunkan,
        # kalau inferensi langsung ngirim nilai jadi (mis. dari form web) -> dipakai.
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Months"] = df["Credit_History_Age"].apply(
                self._credit_history_to_months)
        elif "Credit_History_Months" not in df.columns:
            df["Credit_History_Months"] = np.nan

        if "Type_of_Loan" in df.columns:
            df["Num_Loan_Types"] = df["Type_of_Loan"].apply(self._count_loan_types)
        elif "Num_Loan_Types" not in df.columns:
            df["Num_Loan_Types"] = 0

        # outlier -> NaN biar nanti diimpute median
        for c, (lo, hi) in self.OUTLIER_BOUNDS.items():
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan

        df = df.drop(columns=self.DROP_COLS + ["Type_of_Loan", "Credit_History_Age"],
                     errors="ignore")
        return df

    def _split_xy(self, df, training):
        df = self._clean(df)
        y = None
        if training:
            y = df["Credit_Score"]
        df = df.drop(columns=["Credit_Score"], errors="ignore")
        return df, y

    def fit(self, df):
        X, y = self._split_xy(df, training=True)

        self.numeric_cols = [c for c in X.columns if c not in self.CATEGORICAL]
        cat_cols = [c for c in self.CATEGORICAL if c in X.columns]

        numeric_pipe = SkPipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        cat_pipe = SkPipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        self.column_transformer = ColumnTransformer([
            ("num", numeric_pipe, self.numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ])
        self.column_transformer.fit(X)
        self.feature_names = list(X.columns)
        self.label_encoder.fit(y)
        self.fitted = True
        return self

    def transform(self, df):
        if not self.fitted:
            raise RuntimeError("Preprocessor belum di-fit.")
        X, _ = self._split_xy(df, training=False)
        # samakan kolom dgn waktu fit (kalau inferensi cuma kirim sebagian)
        X = X.reindex(columns=self.feature_names)
        return self.column_transformer.transform(X)

    def fit_transform(self, df):
        self.fit(df)
        X, y = self._split_xy(df, training=True)
        X = X.reindex(columns=self.feature_names)
        Xt = self.column_transformer.transform(X)
        yt = self.label_encoder.transform(y)
        return Xt, yt

    def encode_target(self, y):
        return self.label_encoder.transform(y)

    def decode_target(self, y):
        return self.label_encoder.inverse_transform(y)


# ---------------------------------------------------------------------------
# 2. EVALUATION
# ---------------------------------------------------------------------------
class ModelEvaluator:
    """Hitung metrik klasifikasi. Dipisah biar gampang dipanggil ulang."""

    @staticmethod
    def evaluate(y_true, y_pred):
        from sklearn.metrics import (accuracy_score, f1_score,
                                     precision_score, recall_score)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
        }

    @staticmethod
    def report(y_true, y_pred, target_names=None):
        from sklearn.metrics import classification_report, confusion_matrix
        print(classification_report(y_true, y_pred, target_names=target_names))
        print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


# ---------------------------------------------------------------------------
# 3. TRAINING
# ---------------------------------------------------------------------------
class ModelTrainer:
    """
    Nampung kandidat model dan training-nya. Tiap model dicatat ke MLflow
    (params + metrik test). Yang terbaik (pakai f1_macro karena kelasnya
    imbalance) disimpan.
    """

    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.results = {}        # nama -> {model, metrics}
        self.evaluator = ModelEvaluator()

    def _candidates(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced",
                random_state=self.random_state),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=16, min_samples_leaf=8, n_jobs=-1,
                class_weight="balanced", random_state=self.random_state),
            "xgboost": XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
                random_state=self.random_state, n_jobs=-1),
        }

    def train_all(self, X_train, y_train, X_test, y_test, target_names=None):
        import mlflow

        mlflow.set_experiment("credit_score_classification")
        for name, model in self._candidates().items():
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                metrics = self.evaluator.evaluate(y_test, pred)

                mlflow.log_param("model_type", name)
                mlflow.log_params(
                    {k: v for k, v in model.get_params().items()
                     if isinstance(v, (int, float, str, bool)) and v is not None})
                mlflow.log_metrics(metrics)

                self.results[name] = {"model": model, "metrics": metrics}
                print(f"[{name}] " + " | ".join(
                    f"{k}={v:.4f}" for k, v in metrics.items()))
        return self.results

    def best(self, metric="f1_macro"):
        name = max(self.results, key=lambda n: self.results[n]["metrics"][metric])
        return name, self.results[name]["model"], self.results[name]["metrics"]


# ---------------------------------------------------------------------------
# 4. ORKESTRASI
# ---------------------------------------------------------------------------
class CreditScorePipeline:
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.pre = DataPreprocessor()
        self.trainer = ModelTrainer()

    def run(self):
        df = pd.read_csv(self.data_path, index_col=0)
        print(f"Data: {df.shape[0]} baris, {df.shape[1]} kolom")

        X, y = self.pre.fit_transform(df)
        target_names = list(self.pre.label_encoder.classes_)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

        self.trainer.train_all(X_train, y_train, X_test, y_test, target_names)

        name, model, metrics = self.trainer.best()
        print(f"\n>> Model terbaik: {name} (f1_macro={metrics['f1_macro']:.4f})")
        print("\nDetail model terbaik di data test:")
        self.trainer.evaluator.report(
            y_test, model.predict(X_test), target_names=target_names)

        self._save(model, name, metrics, target_names)
        return name, metrics

    def _save(self, model, name, metrics, target_names):
        MODEL_DIR.mkdir(exist_ok=True)
        with open(MODEL_DIR / "best_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(MODEL_DIR / "preprocessor.pkl", "wb") as f:
            pickle.dump(self.pre, f)
        meta = {
            "best_model": name,
            "metrics": metrics,
            "classes": target_names,
            "features": self.pre.feature_names,
        }
        with open(MODEL_DIR / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nArtefak tersimpan di folder '{MODEL_DIR}/'")


if __name__ == "__main__":
    # Import lewat nama modul "pipeline" (bukan __main__) supaya kelas
    # DataPreprocessor ke-pickle sebagai 'pipeline.DataPreprocessor'.
    # Kalau dipickle dari __main__, inference.py/app.py gagal load modelnya.
    from pipeline import CreditScorePipeline as _Pipeline
    _Pipeline().run()
