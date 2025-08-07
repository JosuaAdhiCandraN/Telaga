import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Dummy data
data = {
    "tanggal": pd.date_range(start="2025-07-19", periods=13, freq="D"),
    "jumlah_pengunjung": [475, 590, 316, 106, 266, 147, 472, 346, 398, 428, 465, 556, 352]
}
df = pd.DataFrame(data)
df.set_index('tanggal', inplace=True)

# Fit model ARIMA
model = ARIMA(df['jumlah_pengunjung'], order=(2, 1, 2))
model_fit = model.fit()

# Forecast 7 hari ke depan
forecast = model_fit.forecast(steps=7)
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)
forecast_df = pd.DataFrame({
    'tanggal': future_dates,
    'prediksi_jumlah_pengunjung': forecast.round().astype(int)
}).set_index('tanggal')

# Tampilkan
print("\n📈 Prediksi Pengunjung:\n")
print(forecast_df)

# Visualisasi
plt.figure(figsize=(10,5))
plt.plot(df.index, df['jumlah_pengunjung'], label='Aktual')
plt.plot(forecast_df.index, forecast_df['prediksi_jumlah_pengunjung'], label='Prediksi', linestyle='--')
plt.title("Prediksi Jumlah Pengunjung (ARIMA)")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah Pengunjung")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
