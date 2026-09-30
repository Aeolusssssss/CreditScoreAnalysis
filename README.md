# Credit Score Classification — Model Deployment

Proyek menilai performa kredit nasabah (`Good` / `Standard` / `Poor`) menggunakan
machine learning, lengkap dengan pipeline training (OOP + MLflow), inferencing, dan
deployment berbasis web (Streamlit).

## Struktur folder

```
.
├── 1_eksplorasi_modelling.ipynb   # eksplorasi data + eksperimen model
├── pipeline.py                    # pipeline training OOP + MLflow (retraining)
├── inference.py                   # load model + prediksi (dipakai app & test)
├── app.py                         # web app Streamlit
├── requirements.txt
├── data_A.csv                     # dataset (sesuai ketentuan NIM)
├── model/
│   ├── best_model.pkl             # model terbaik (XGBoost)
│   ├── preprocessor.pkl           # preprocessing fitted (cleaning + encoder)
│   └── metadata.json              # info model & metrik
├── screenshot_test_deployment.pdf # bukti test case di app
├── PANDUAN_AWS.md                 # langkah deploy ke AWS
├── PERBANDINGAN_local_vs_cloud.md # analisis local vs cloud
└── SKRIP_VIDEO.md                 # skrip narasi video penjelasan
```

## Cara menjalankan (local)

1. Buat virtual environment & install dependency:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. (Opsional) Training ulang model. Butuh `mlflow` (lihat catatan di requirements.txt):
   ```bash
   pip install mlflow
   python pipeline.py
   ```
   Hasil eksperimen bisa dilihat dengan:
   ```bash
   mlflow ui            # buka http://localhost:5000
   ```

3. Cek inferencing lewat test case:
   ```bash
   python inference.py
   ```

4. Jalankan web app:
   ```bash
   streamlit run app.py
   ```
   Buka `http://localhost:8501`. Di sidebar ada preset test case (Good/Standard/Poor)
   untuk demo cepat.

## Ringkasan model

| Model               | Accuracy | F1-macro |
|---------------------|----------|----------|
| Logistic Regression | 0.648    | 0.641    |
| Random Forest       | 0.696    | 0.687    |
| **XGBoost** (dipilih) | **0.739** | **0.715** |

Metrik utama yang dipakai adalah **F1-macro** karena target tidak seimbang
(`Standard` mendominasi). XGBoost dipilih karena F1-macro tertinggi sekaligus
ukuran file model kecil (~4.5 MB), cocok untuk deployment.
