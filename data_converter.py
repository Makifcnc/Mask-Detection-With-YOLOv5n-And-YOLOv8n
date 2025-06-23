import os
import xml.etree.ElementTree as ET
from PIL import Image

def convert_xml_to_yolo(xml_path, image_path, output_dir):
    """XML annotation dosyasını YOLO formatına dönüştürür"""
    
    # XML dosyasını parse et
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Resim boyutlarını al
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Sınıf isimlerini tanımla
    class_names = ['with_mask', 'without_mask', 'mask_weared_incorrect']
    
    yolo_lines = []
    
    # Her object için bounding box'ı dönüştür
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name in class_names:
            class_id = class_names.index(class_name)
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # YOLO formatına dönüştür (center_x, center_y, width, height)
            center_x = (xmin + xmax) / 2.0 / img_width
            center_y = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines

def main():
    """Ana dönüştürme fonksiyonu"""
    
    # Dizinleri oluştur
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)
    
    # XML dosyalarını listele
    xml_dir = 'maske-dataset/annotations'
    image_dir = 'maske-dataset/images'
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    # %80 train, %20 validation olarak böl
    train_count = int(len(xml_files) * 0.8)
    
    for i, xml_file in enumerate(xml_files):
        xml_path = os.path.join(xml_dir, xml_file)
        image_file = xml_file.replace('.xml', '.png')
        image_path = os.path.join(image_dir, image_file)
        
        # Resim dosyası var mı kontrol et
        if not os.path.exists(image_path):
            continue
        
        # YOLO formatına dönüştür
        yolo_lines = convert_xml_to_yolo(xml_path, image_path, 'dataset')
        
        # Train veya validation olarak ayır
        if i < train_count:
            dest_img_dir = 'dataset/images/train'
            dest_label_dir = 'dataset/labels/train'
        else:
            dest_img_dir = 'dataset/images/val'
            dest_label_dir = 'dataset/labels/val'
        
        # Resmi kopyala
        import shutil
        shutil.copy2(image_path, os.path.join(dest_img_dir, image_file))
        
        # Label dosyasını oluştur
        label_file = xml_file.replace('.xml', '.txt')
        with open(os.path.join(dest_label_dir, label_file), 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    print(f"Toplam {len(xml_files)} dosya dönüştürüldü!")
    print(f"Train: {train_count}, Validation: {len(xml_files) - train_count}")

if __name__ == "__main__":
    main() 