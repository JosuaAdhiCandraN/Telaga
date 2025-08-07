import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from prefect import flow, task

# === TASKS ===

@task
def generate_dummy_data():
    np.random.seed(42)
    n = 100

    data = pd.DataFrame({
        'tanggal': pd.date_range(start='2025-01-01', periods=n, freq='D'),
        'kapasitas_terpakai': np.random.randint(100, 1000, size=n),
        'jumlah_pengunjung': np.random.randint(50, 900, size=n)
    })
    print("📦 Dummy data generated")
    return data

@task
def preprocess(df: pd.DataFrame):
    df = df.dropna()
    print("🧼 Data cleaned")
    return df

@task
def train_model(df: pd.DataFrame):
    X = df[['kapasitas_terpakai']]
    y = df['jumlah_pengunjung']

    model = LinearRegression()
    model.fit(X, y)

    print("📈 Model trained")
    return model

@task
def predict(df: pd.DataFrame, model):
    df['prediksi_pengunjung'] = model.predict(df[['kapasitas_terpakai']])
    print("🔮 Prediction done")
    return df

@task
def store_output(df: pd.DataFrame):
    df.to_csv("prediksi_pengunjung_summary.csv", index=False)
    print("💾 Output stored to prediksi_pengunjung_summary.csv")

# === FLOW ===

@flow(name="HOLAP - Prefect Prototype")
def holap_pipeline():
    data = generate_dummy_data()
    clean_data = preprocess(data)
    model = train_model(clean_data)
    prediction = predict(clean_data, model)
    store_output(prediction)

# Run the pipeline
if __name__ == "__main__":
    holap_pipeline()
