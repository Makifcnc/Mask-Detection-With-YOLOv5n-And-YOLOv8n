import cv2
import numpy as np
from ultralytics import YOLO
import time

class MaskDetector:
    def __init__(self, model_path='mask_detection/mask_model3/weights/best.pt'):
        """Maske tespit sınıfını başlatır"""
        self.model = YOLO(model_path)
        self.class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
        self.colors = [(0, 255, 0), (0, 0, 255), (255, 165, 0)]  # Yeşil, Kırmızı, Turuncu
        
    def get_mask_status_text(self, class_id):
        """Sınıf ID'sine göre Türkçe durum metni döndürür"""
        status_texts = {
            0: "Maske Takilmis ",
            1: "Maske Takilmamis ", 
            2: "Maske Yanlis Takilmis "
        }
        return status_texts.get(class_id, "Bilinmeyen")
    
    def detect_masks(self, frame):
        """Frame'de maske tespiti yapar"""
        results = self.model(frame, conf=0.5)  # %50 güven eşiği
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Bounding box koordinatları
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Sınıf ve güven skoru
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Renk ve metin
                    color = self.colors[class_id]
                    status_text = self.get_mask_status_text(class_id)
                    
                    # Bounding box çiz
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Metin arka planı
                    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1-30), (x1 + text_size[0], y1), color, -1)
                    
                    # Metin yaz
                    cv2.putText(frame, status_text, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Güven skoru
                    conf_text = f"Confidence Ratio: {confidence:.2f}"
                    cv2.putText(frame, conf_text, (x1, y2+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run_camera(self):
        """Kamera ile gerçek zamanlı tespit yapar"""
        cap = cv2.VideoCapture(0)  # Varsayılan kamera
        
        if not cap.isOpened():
            print("Kamera acilamadi!")
            return
        
        print("Maske Tespit Sistemi Baslatildi!")
        print("Cikmak icin 'q' tusuna basin")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame okunamadi!")
                break
            
            # Maske tespiti yap
            processed_frame = self.detect_masks(frame)
            
            # FPS hesapla
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Ekrana göster
            cv2.imshow('Maske Tespit Sistemi', processed_frame)
            
            # 'q' tuşu ile çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Ana fonksiyon"""
    try:
        detector = MaskDetector()
        detector.run_camera()
    except Exception as e:
        print(f"Hata olustu: {e}")
        print("Model dosyasinin dogru yolda oldugundan emin olun!")

if __name__ == "__main__":
    main() 