# Implementasi Self-Supervised Contrastive Learning untuk Klasifikasi Pneumonia pada Citra X Ray Thorax dengan Ketersediaan Label Terbatas

Repositori ini berisi keseluruhan kode sumber (source code) dari penelitian tugas akhir (skripsi) untuk memenuhi sebagian persyaratan akademik.

**Informasi Peneliti**:

- **Nama**: Kirouch Alqornie Gymnastiar

- **NIM**: 202210370311189

- **Dosen Pembimbing**: Yuda Munarko, S.Kom., M.Sc

- **Program Studi**: Informatika

- **Fakultas**: Teknik

- **Universitas**: Universitas Muhammadiyah Malang

Penelitian ini berfokus pada penyelesaian masalah kelangkaan label (label scarcity) dan ketidakseimbangan kelas (imbalanced data) pada diagnosis pneumonia dengan fokus demografi pediatrik (anak-anak). Sistem dibangun menggunakan kerangka kerja Self-Supervised Learning (SimCLR) dan tulang punggung ResNet-18 untuk menghasilkan model diagnostik opini kedua (second opinion) yang tangguh, transparan, dan terkalibrasi secara klinis.

## Arsitektur Data: Skenario Dua Dataset (Cross-Domain)
Penelitian ini mengadopsi skenario transfer pembelajaran lintas domain yang memanfaatkan dua himpunan data independen dari Kaggle:

1. **Domain Sumber (Pre-training): [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/)**
   * **Karakteristik:** Citra rontgen dada pasien dewasa berjumlah ~26.684 data.
   * **Peran:** Digunakan pada fase Self-Supervised Learning (SimCLR). Model dilatih menggunakan augmentasi kontrastif untuk mempelajari ekstraksi fitur intrinsik paru-paru secara mandiri tanpa campur tangan anotasi dari dokter radiologi.
2. **Domain Target (Fine-Tuning): [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**
   * **Karakteristik:** Citra rontgen dada pediatrik (anak-anak usia 1-5 tahun) berjumlah 5.856 citra dengan distribusi kelas yang sangat timpang (27% Normal vs 73% Pneumonia).
   * **Peran:** igunakan pada fase _Supervised Fine-Tuning_. Klasifikasi disesuaikan secara khusus untuk demografi pediatrik guna memvalidasi kemampuan adaptasi fitur (transferability) model pada data berdimensi medis spesifik.

## Sorotan Metodologi Utama

Bagian ini menyoroti berbagai inovasi dan penyesuaian arsitektur komputasi yang dirancang khusus untuk mengatasi tantangan klasik pada pemrosesan citra medis. Seluruh pendekatan di bawah ini difokuskan untuk memastikan model tidak hanya memiliki akurasi yang tinggi, tetapi juga aman, transparan, dan dapat dipertanggungjawabkan secara klinis:

- Anatomy-Preserving Transform: Modifikasi pipa augmentasi (menggantikan RandomResizedCrop konvensional dengan Affine ringan) untuk mencegah terpotongnya organ vital (jantung dan batas tulang rusuk) selama pembuatan Positive Pairs di fase SimCLR.

- Mitigasi Imbalance (Single Penalty): Penggunaan WeightedRandomSampler pada Dataloader dipadukan dengan Dropout (50%) dan Weight Decay (1e-2) untuk mencegah model AI menghafal data kelas mayoritas (overfitting).

- Optimal Decision Threshold (Youden's J Statistic): Kalibrasi ambang batas probabilitas secara matematis berbasis kurva ROC untuk menekan angka False Positives (alarm palsu pada anak sehat) hingga batas minimum tanpa mengorbankan sensitivitas infeksi.

- Explainable AI (Grad-CAM++): Audit visual yang membuktikan bahwa model secara akurat mengunci atensi pada infiltrat patologis (bercak cairan), bukan pada artefak anatomis bawaan mesin.

- Analisis Ruang Laten (t-SNE): Pemetaan dua dimensi yang memvalidasi efektivitas arsitektur usulan dalam memisahkan klaster rontgen sehat dan sakit dibandingkan arsitektur baseline konvensional.

## Infrastruktur Komputasi & Instruksi Eksekusi

Mengingat ukuran dataset yang masif (lebih dari 1 GB total), dataset mentah dan beban bobot model akhir (.pth) tidak diikutsertakan di dalam repositori ini sesuai kebijakan batasan GitHub. Namun, alur komputasi (I/O) di dalam notebook telah dirancang secara efisien untuk dieksekusi di cloud.

**Langkah 1: Preprocessing Lokal (VSCode / Terminal)**

Untuk menghindari kehabisan memori (OOM) pada Google Colab, pemrosesan awal representasi piksel Dataset RSNA dilakukan secara lokal.

1. Unduh dataset RSNA dari Kaggle ke mesin lokal Anda.

2. Jalankan skrip 00_preprocessing_rsna_lokal.py. Kode ini akan mengekstrak file DICOM mentah ke .jpg, menstandarisasi resolusi menjadi 224x224, dan mengompresnya menjadi rsna_preprocessed.zip.

3. Unggah rsna_preprocessed.zip beserta arsip asli dataset Chest X-Ray Pediatric ke dalam Google Drive Anda.

**Langkah 2: Eksekusi Google Colab (Mulai dari Cell 1)**

Buka file notebook utama (RSNA PNEUMONIA 14 JULI.ipynb) di Google Colab. Kode dirancang dengan arsitektur I/O terakselerasi:

- Autentikasi & Koneksi: Notebook akan meminta izin mounting ke Google Drive Anda.

- Akselerasi I/O Lokal (Krusial): Sistem dirancang agar tidak membaca citra rontgen secara langsung dari Google Drive untuk menghindari bottleneck latensi jaringan. Kode akan menyalin dan mengekstrak .zip secara dinamis ke penyimpanan lokal instans Colab (NVMe SSD storage). Teknik ini menghemat waktu siklus training secara signifikan.

- Konfigurasi Global: Eksekusi Cell 1 untuk menginisialisasi parameter komputasi akselerator perangkat keras (GPU / CUDA).

**Langkah 3: Pelatihan dan Evaluasi (End-to-End)**

Setelah Cell 1 tuntas dieksekusi tanpa galat, jalankan sisa cell secara berurutan. Pipa komputasi akan secara otomatis memproses:

1. Inspeksi Visual Data (EDA).

2. Pre-training Self-Supervised (SimCLR).

3. Pemindahan Pengetahuan (Transfer Learning & Fine-Tuning).

4. Pembuatan Matriks Kebingungan (Confusion Matrix) dan Kalibrasi Threshold.

5. Render Visualisasi Grad-CAM++ dan Grafik t-SNE.
