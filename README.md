# Klasifikasi Pneumonia Pediatrik Menggunakan Self-Supervised Contrastive Learning pada Label Terbatas

Repositori ini berisi keseluruhan kode sumber (*source code*) dari penelitian tugas akhir yang disusun oleh Kirouch Alqornie Gymnastiar. Penelitian ini berfokus pada penyelesaian masalah kelangkaan label (*label scarcity*) dan ketidakseimbangan kelas (*imbalanced data*) pada diagnosis pneumonia anak menggunakan arsitektur **SimCLR** dan **ResNet-18**.

## 📊 Arsitektur Data: Skenario Dua Dataset (Cross-Domain)
Penelitian ini mengadopsi skenario transfer pembelajaran lintas domain yang memanfaatkan dua himpunan data independen dari Kaggle:

1. **Domain Sumber (Pre-training): [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/)**
   * **Karakteristik:** Citra rontgen dada pasien dewasa berjumlah ~26.684 data.
   * **Peran:** Digunakan sebagai dataset *unlabeled* pada fase *Self-Supervised Learning* (SimCLR). Model dilatih menggunakan augmentasi kontrastif untuk mempelajari ekstraksi fitur intrinsik paru-paru secara mandiri tanpa campur tangan anotasi medis.
2. **Domain Target (Fine-Tuning): [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)**
   * **Karakteristik:** Citra rontgen dada pediatrik (anak-anak usia 1-5 tahun) berjumlah 5.856 citra dengan distribusi kelas yang sangat timpang (27% Normal vs 73% Pneumonia).
   * **Peran:** Digunakan pada fase *Supervised Fine-Tuning*. Klasifikasi akhir disesuaikan menggunakan *Weighted Random Sampler* untuk memitigasi bias prediksi ke kelas mayoritas.

## 🚀 Infrastruktur Komputasi & Instruksi Eksekusi

Mengingat ukuran dataset yang masif (lebih dari 100MB), dataset dan bobot model (`.pt`/`.pth`) **tidak disertakan** di dalam repositori ini sesuai kebijakan batasan limit GitHub. Namun, alur I/O telah dirancang sangat efisien. 

### Langkah 1: Preprocessing Lokal (VSCode)
Untuk menghindari batasan memori (RAM/Disk) pada Google Colab, pemrosesan awal Dataset RSNA dilakukan secara manual di komputer lokal.
1. Unduh dataset RSNA dari Kaggle ke mesin lokal Anda.
2. Jalankan skrip `00_preprocessing_rsna_lokal.py`. Kode ini akan mengonversi *file* mentah DICOM menjadi `.jpg`, mengubah ukuran menjadi 224x224, dan mengompresnya menjadi `rsna_preprocessed.zip`.
3. Unggah `rsna_preprocessed.zip` beserta arsip dataset Chest X-Ray Pediatric ke dalam **Google Drive** Anda.

### Langkah 2: Eksekusi Google Colab (Mulai dari Cell 1)
Buka *file* Notebook utama (`RSNA_PNEUMONIA.ipynb`) di Google Colab. Kode dirancang dengan arsitektur I/O terakselerasi:
* **Autentikasi & Koneksi:** Notebook akan meminta izin *mounting* ke Google Drive Anda.
* **Akselerasi I/O (Penting!):** Sistem tidak membaca citra secara langsung dari Google Drive (yang akan memicu *bottleneck* latensi jaringan parah). Kode secara otomatis akan memindahkan dan mengekstrak `.zip` dari Drive ke penyimpanan lokal instans Colab (NVMe SSD). Teknik ini mempercepat siklus *training* hingga 10x lipat.
* **Konfigurasi Global:** Sel *setup* awal (Cell 1) juga akan menginisialisasi akselerator perangkat keras (GPU T4/CUDA) dan *hyperparameters* model.

### Langkah 3: Pelatihan dan Evaluasi
Setelah *Cell 1* tuntas, jalankan keseluruhan *cell* secara berurutan (*Run All*). Kode akan mengeksekusi:
1. Pelatihan SimCLR (Fase Kontrastif) menggunakan fungsi *NT-Xent Loss*.
2. Klasifikasi *Fine-tuning* biner dengan metrik keseimbangan data.
3. Evaluasi Kinerja (Akurasi, Presisi, Sensitivitas, Spesifisitas, F1-Score).
4. Audit Diagnostik Visual menggunakan algoritma **Grad-CAM++** untuk membuktikan bahwa atensi jaringan fokus pada infiltrat, bukan pada artefak anatomis tumpang tindih tulang rusuk anak.
