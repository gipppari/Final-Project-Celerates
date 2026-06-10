# Customer Behavior Analytics & Recommendation System

Aplikasi ini sudah dikonversi menjadi aplikasi Python berbasis Streamlit untuk memenuhi ketentuan final project. Aplikasi menampilkan dashboard e-commerce, segmentasi pelanggan RFM, profil pelanggan, rekomendasi produk, analisis tren penjualan, forecast sederhana, dan AI insights.

## Fitur

- Executive dashboard untuk KPI revenue, order, customer, AOV, loyalty rate, dan growth.
- Filter global berdasarkan periode, kategori, segment, gender, device, dan income level.
- Customer segmentation menggunakan RFM scoring.
- Customer profile explorer dan rekomendasi produk berbasis kategori favorit, sensitivitas diskon, dan income level.
- Sales trend analysis berdasarkan bulan, hari, dan simple 30-day forecast.
- AI insights memakai Gemini jika `GEMINI_API_KEY` tersedia di Streamlit secrets, dengan fallback heuristic jika tidak ada API key.

## Run Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Cloud

1. Upload folder project ini ke GitHub.
2. Buka Streamlit Cloud dan pilih repository tersebut.
3. Set main file path ke:

```text
app.py
```

4. Jika ingin memakai Gemini AI, tambahkan secret berikut di Streamlit Cloud:

```toml
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
```

Tanpa secret tersebut, aplikasi tetap berjalan memakai local heuristic insights.

## Dataset

Aplikasi mengambil dataset CSV dari Google Sheets yang sama dengan aplikasi awal. Jika dataset online gagal dimuat, pengguna bisa upload file CSV lewat sidebar.
