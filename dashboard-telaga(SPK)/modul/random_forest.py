import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === FITUR 4: Prediksi Kelayakan Air ===
def prediksi_kelayakan_air():
    data = {
        "ph": [6.8, 7.1, 6.5, 7.3, 8.0, 7.8, 6.9, 7.2, 7.4, 6.7],
        "suhu": [27, 28, 30, 26, 25, 29, 28, 27, 26, 30],
        "kejernihan": [1.1, 1.3, 2.5, 1.0, 0.9, 1.2, 1.5, 1.0, 1.1, 2.2],
        "label": ["layak", "layak", "tidak", "layak", "layak", "layak", "tidak", "layak", "layak", "tidak"]
    }

    df = pd.DataFrame(data)
    X = df[["ph", "suhu", "kejernihan"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n🧪 Evaluasi Model Kelayakan Air:")
    print(classification_report(y_test, y_pred))

    sample = pd.DataFrame({"ph": [7.1], "suhu": [27.4], "kejernihan": [1.2]})
    hasil = model.predict(sample)[0]
    print(f"\n🌊 Prediksi Kelayakan untuk sampel: {hasil}")

if __name__ == "__main__":
    prediksi_kelayakan_air()