import os
import shutil
import random
import torch
from ultralytics import YOLO

def main():
    base_dir = r"c:\Users\Mariana\Desktop\Delfín"
    imagenes_dir = os.path.join(base_dir, "Detección y Clasificación de Humedad Activa y Restos de Humedad", "Imagenes")
    folder_eflo = os.path.join(imagenes_dir, "eflorescensia")
    
    if not os.path.exists(folder_eflo):
        print(f"Error: No se encontró la carpeta {folder_eflo}.")
        exit(1)
        
    # Directorio temporal aislado en descargas para este entrenamiento de eflorescencia
    dataset_dir = r"C:\Users\Mariana\Downloads\modelo-humedad-fisuras-main\temp_train_dataset_eflorescencia"
    yaml_path = r"C:\Users\Mariana\Downloads\modelo-humedad-fisuras-main\temp_data_eflorescencia.yaml"
    
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
        
    os.makedirs(os.path.join(dataset_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "val"), exist_ok=True)
    
    # 1. Recolectar pares de (imagen, txt) únicamente de la carpeta 'eflorescensia'
    labeled_pairs = []
    
    for file_name in os.listdir(folder_eflo):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(file_name)[0]
            txt_name = name + ".txt"
            txt_path = os.path.join(folder_eflo, txt_name)
            
            if os.path.exists(txt_path) and txt_name != "classes.txt":
                with open(txt_path, 'r') as f:
                    content = f.read().strip()
                if content:
                    labeled_pairs.append((os.path.join(folder_eflo, file_name), txt_path))
                    
    print(f"Total de imágenes de eflorescencia recopiladas: {len(labeled_pairs)}")
    
    if len(labeled_pairs) < 5:
        print("Error: Cantidad insuficiente de datos para entrenar.")
        exit(1)
        
    # Mezclar y dividir en train/val (80/20)
    random.seed(42)
    random.shuffle(labeled_pairs)
    split_idx = int(len(labeled_pairs) * 0.8)
    train_pairs = labeled_pairs[:split_idx]
    val_pairs = labeled_pairs[split_idx:]
    
    print(f"División: {len(train_pairs)} para Entrenamiento, {len(val_pairs)} para Validación.")
    
    # Copiar archivos tal cual — los labels ya tienen el índice 1 (Eflorescencia)
    # nc=2 en el YAML permite que el índice 1 sea válido sin conversión
    def copy_pairs(pairs, subset):
        for img_path, txt_path in pairs:
            base_name = os.path.basename(img_path)
            name_without_ext, ext = os.path.splitext(base_name)
            shutil.copy(img_path, os.path.join(dataset_dir, "images", subset, base_name))
            shutil.copy(txt_path, os.path.join(dataset_dir, "labels", subset, name_without_ext + ".txt"))
            
    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    
    # 2. YAML con 2 clases: Humedad (índice 0) y Eflorescencia (índice 1)
    dataset_path_formatted = dataset_dir.replace('\\', '/')
    yaml_content = f"""path: {dataset_path_formatted}
train: images/train
val: images/val

nc: 2
names: ['Humedad', 'Eflorescencia']
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    # 3. Configurar hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entrenando en: {device.upper()}")
    
    # 4. Modelo base yolov8n
    model = YOLO("yolov8n.pt")
    
    # 5. Entrenar a 50 épocas
    epochs = 50
    batch_size = 16 if device == "cuda" else 4
    print(f"Iniciando entrenamiento — Humedad=0, Eflorescencia=1, {epochs} épocas...")
    
    model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        device=device,
        project=os.path.join(base_dir, "Detección y Clasificación de Humedad Activa y Restos de Humedad", "runs"),
        name="modelo_eflorescencia_pura",
        exist_ok=True,
        workers=0,
        patience=15,
        flipud=0.5,
        hsv_v=0.6,
        degrees=10.0,
        mixup=0.05
    )
    
    print("\n¡Entrenamiento finalizado con éxito!")
    
    best_weights = os.path.join(base_dir, "Detección y Clasificación de Humedad Activa y Restos de Humedad", "runs", "modelo_eflorescencia_pura", "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"\n[OK] Modelo guardado en:\n{best_weights}")
        
    # 6. Limpieza final
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    if os.path.exists(yaml_path):
        os.remove(yaml_path)

if __name__ == '__main__':
    main()
