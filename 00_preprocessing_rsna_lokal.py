"""
Script Preprocessing Lokal: RSNA Pneumonia Detection Challenge
Tujuan: Menghemat I/O dan memori Google Colab dengan melakukan ekstraksi, 
resizing (224x224), dan konversi format DICOM ke JPG secara lokal, 
lalu mengompresnya menjadi file ZIP (rsna_preprocessed.zip).
"""

import os
import cv2
import pydicom
import zipfile
import numpy as np
from tqdm import tqdm

# 1. Konfigurasi Path (Menggunakan raw string 'r' untuk Windows)
RAW_DICOM_DIR = r'C:\Users\Kirouch Alqornie Gym\Documents\Kuliah\Semester 8\Skripsi_Pneumonia\rsna_raw_data\RSNA Pneumonia Detection Challenge\stage_2_train_images' 
OUTPUT_DIR = r'C:\Users\Kirouch Alqornie Gym\Documents\Kuliah\Semester 8\Skripsi_Pneumonia\rsna_preprocessed_224'
ZIP_FILENAME = r'C:\Users\Kirouch Alqornie Gym\Documents\Kuliah\Semester 8\Skripsi_Pneumonia\rsna_preprocessed.zip'
TARGET_SIZE = (224, 224)

def preprocess_and_zip():
    # Buat folder output jika belum ada
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    dicom_files = [f for f in os.listdir(RAW_DICOM_DIR) if f.endswith('.dcm')]
    print(f"Total gambar ditemukan: {len(dicom_files)}")
    
    # 2. Proses Konversi dan Resizing
    for filename in tqdm(dicom_files, desc="Processing DICOM to JPG"):
        filepath = os.path.join(RAW_DICOM_DIR, filename)
        
        # Baca file medis DICOM
        dicom = pydicom.dcmread(filepath)
        img = dicom.pixel_array
        
        # Normalisasi ke skala 0-255 (Grayscale)
        img = img - np.min(img)
        img = (img / np.max(img) * 255).astype(np.uint8)
        
        # Resize ke 224x224
        img_resized = cv2.resize(img, TARGET_SIZE)
        
        # Simpan sebagai JPG
        jpg_filename = filename.replace('.dcm', '.jpg')
        cv2.imwrite(os.path.join(OUTPUT_DIR, jpg_filename), img_resized)
        
    # 3. Kompresi menjadi ZIP (Untuk di-upload ke Google Drive)
    print(f"\nMengompres folder {OUTPUT_DIR} menjadi {ZIP_FILENAME}...")
    with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in tqdm(files, desc="Zipping files"):
                zipf.write(os.path.join(root, file), arcname=file)
                
    print("\n✅ Preprocessing selesai! Silakan upload 'rsna_preprocessed.zip' ke Google Drive.")

if __name__ == '__main__':
    preprocess_and_zip()