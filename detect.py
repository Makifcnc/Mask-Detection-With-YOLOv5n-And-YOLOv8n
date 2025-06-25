import cv2
import torch
import yaml
import numpy as np

# Sınıf isimlerini dataset.yaml'dan oku
with open('dataset.yaml', 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)
    class_names = [data['names'][i] for i in range(data['nc'])]

# Eğitilmiş YOLOv5 modelini yükle
model = torch.hub.load('yolov5', 'custom', path='mask_detection/mask_model_v5/weights/last.pt', source='local')

# Renkler (her sınıf için farklı)
COLORS = [(0, 255, 0), (0, 0, 255), (255, 165, 0)]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Kamera acilamadi!')
    exit()

print('Egitilmis YOLOv5 Modeli ile Maske Tespit Sistemi Baslatildi!')
print("Cikmak icin 'q' tusuna basin")

while True:
    ret, frame = cap.read()
    if not ret:
        print('Frame okunamadi!')
        break

    # YOLOv5 ile tahmin
    results = model(frame)
    preds = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    for *box, conf, cls_id in preds:
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls_id)
        color = COLORS[class_id % len(COLORS)]
        label = f"{class_names[class_id]} ({conf:.2f})"

        # Kutu çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Etiket arka planı
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-30), (x1+tw, y1), color, -1)
        # Etiket yaz
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow('Eğitilmiş YOLOv5 Maske Tespit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 