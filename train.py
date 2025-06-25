import os
import sys
import subprocess

# YOLOv5 klasörüne git
os.chdir('yolov5')

# Eğitim komutunu hazırla
command = [
    sys.executable, 'train.py',
    '--img', '640',
    '--batch', '16',
    '--epochs', '50',
    '--data', '../dataset.yaml',
    '--weights', 'yolov5n.pt',
    '--project', '../mask_detection',
    '--name', 'mask_model_v5',
    '--exist-ok'
]

# Eğitimi başlat
subprocess.run(command) 