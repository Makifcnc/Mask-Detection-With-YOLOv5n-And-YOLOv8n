from ultralytics import YOLO
import os

def train_mask_detection_model():
    """Maske tespit modelini eğitir"""
    
    # YOLO modelini yükle (küçük model - hızlı eğitim için)
    model = YOLO('yolov8n.pt')
    
    # Model eğitimi
    print("Model eğitimi başlıyor...")
    results = model.train(
        data='dataset.yaml',
        epochs=50,  # Epoch sayısı
        imgsz=640,  # Resim boyutu
        batch=16,   # Batch size
        device='cpu',  # CPU kullan (GPU varsa '0' yapabilirsiniz)
        patience=10,   # Early stopping
        save=True,
        project='mask_detection',
        name='mask_model'
    )
    
    print("Model eğitimi tamamlandı!")
    print(f"Model kaydedildi: mask_detection/mask_model/weights/best.pt")

if __name__ == "__main__":
    train_mask_detection_model() 