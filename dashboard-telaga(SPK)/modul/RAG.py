import os
from openai import OpenAI

# === Setup client ===
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise EnvironmentError("❌ Environment variable OPENROUTER_API_KEY tidak ditemukan.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# === Prompt Dummy Data ===
prompt = """
Berdasarkan data tempat wisata berikut:

- Jumlah kunjungan 7 hari terakhir: 1200, 1900, 2100, 1150, 1000, 950, 920
- Komposisi pengunjung: 52% keluarga, 30% remaja, 18% lansia
- Ulasan: positif (kebersihan, staf), negatif (parkir, antrian)
- Kualitas air: pH 7.1, suhu 27.4°C, kejernihan 1.2 NTU (aman)

Langkah:
1. Analisis tren kunjungan dan karakteristik pengunjung.
2. Identifikasi dua masalah utama dari ulasan.
3. Buat 5 rekomendasi operasional realistis untuk meningkatkan layanan minggu ini.

Tulis hasilnya secara profesional dan praktis.
"""

# === Kirim ke model ===
def run_spk_ai():
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1-0528:free",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    result = response.choices[0].message.content.strip()

    print("\n📊 === HASIL ANALISIS TEMPAT WISATA ===\n")
    print(result)

# === Eksekusi ===
if __name__ == "__main__":
    run_spk_ai()
