from ultralytics import YOLO
import os

def train_mask_detection_model():
    """Maske tespit modelini eğitir"""
    
    # YOLO modelini yükle (küçük model - hızlı eğitim icin)
    model = YOLO('yolov8n.pt')
    
  
    print("Model eğitimi başlıyor...")
    results = model.train(
        data='dataset.yaml',
        epochs=50,  
        imgsz=640,  
        batch=16,   
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