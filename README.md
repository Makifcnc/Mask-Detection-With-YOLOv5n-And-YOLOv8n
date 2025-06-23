# Maske Tespit Sistemi

Bu proje YOLOv8n kullanarak gerçek zamanlı maske tespiti yapan bir sistemdir.

## Özellikler

- 3 farklı maske durumu tespiti:
  - ✅ Maske Takılmış
  - ❌ Maske Takılmamış  
  - ⚠ Maske Yanlış Takılmış
- Gerçek zamanlı kamera tespiti


## Veri Seti: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection/data uploaded by https://www.kaggle.com/andrewmvd/datasets.
(Thanks to Andrew MVD for sharing this valuable dataset with the community.)

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Veri Setini Hazırlama
```bash
python data_converter.py
```
Bu komut XML annotation dosyalarını YOLO formatına dönüştürür.

### 2. Model Eğitimi
```bash
python train_model.py
```
Bu komut YOLO modelini eğitir (eğitim uzunluğu cihaz donanımına göre artabilir).

### 3. Gerçek Zamanlı Tespit
```bash
python mask_detection.py
```
Bu komut kamerayı açar ve gerçek zamanlı maske tespiti yapar.

## Dosya Yapısı

```
├── data_converter.py      # Veri dönüştürme scripti
├── train_model.py         # Model eğitim scripti  
├── mask_detection.py      # Ana uygulama
├── dataset.yaml           # Dataset konfigürasyonu
├── requirements.txt       # Gerekli paketler
├── maske-dataset/         # Orijinal veri seti
└── dataset/              # YOLO formatında veri seti
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

## Kontroller

- **q**: Uygulamadan çık
- Kamera otomatik olarak açılır ve tespit başlar

## Notlar

- Model eğitimi CPU'da yapılır (GPU varsa daha hızlı olur)
- İlk çalıştırmada YOLO modeli otomatik indirilir
- Eğitim süresi veri seti boyutuna göre değişir 