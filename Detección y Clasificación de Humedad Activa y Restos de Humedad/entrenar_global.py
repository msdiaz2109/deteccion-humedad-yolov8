import os
import sys
import shutil
import random
import torch
from ultralytics import YOLO

# ============================================================
#  CONFIGURACIÓN DEL PROCESO DE ENTRENAMIENTO
# ============================================================

# Ruta base del proyecto
BASE_DIR = r"c:\Users\Mariana\Desktop\Delfín"

# Carpeta de imágenes originales (vid_001 a vid_010)
CARPETA_DATOS = os.path.join(BASE_DIR, "Detección y Clasificación de Humedad Activa y Restos de Humedad", "Imagenes")

# Carpeta temporal para entrenamiento en descargas (evita errores por caracteres especiales en el path)
DATASET_DIR = r"C:\Users\Mariana\Downloads\modelo-humedad-fisuras-main\temp_train_dataset"
YAML_PATH = r"C:\Users\Mariana\Downloads\modelo-humedad-fisuras-main\temp_data.yaml"

# Configuración del dataset split (20% val, 80% train)
VAL_SPLIT = 0.2

# Parámetros del modelo
EPOCAS = 50           # Aumentado a 150 épocas (como el original de descargas) para convergencia perfecta
LOTE = 16             # Batch size 16 (óptimo para estabilidad del gradiente en tu RTX 4050 6GB)
IMG_SIZE = 640        # Resolución de entrada
LR = 0.01             # Tasa de aprendizaje inicial

# Modelo base de partida
# NOTA: Se usa yolov8n.pt (genérico) en vez de hum_y_fis_v3 porque ese modelo
# tenía 2 clases (Humedad+Fisura) y causaba detecciones de Fisura no deseadas.
MODELO_BASE = "yolov8n.pt"

# Nombre del proyecto para almacenar los resultados
NOMBRE_PROYECTO = "humedad_detector_global"


def organizar_dataset():
    """
    Recopila las imágenes anotadas del proyecto y las organiza en formato YOLO
    en el directorio de descargas temporal.
    """
    print("=" * 60)
    print(" PASO 1: Organizando dataset global de entrenamiento")
    print("=" * 60)
    
    if os.path.exists(dataset_dir := DATASET_DIR):
        shutil.rmtree(dataset_dir)
        
    os.makedirs(os.path.join(DATASET_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels", "val"), exist_ok=True)

    # Recolectar pares de imágenes y etiquetas
    labeled_pairs = []
    subfolders = [d for d in os.listdir(CARPETA_DATOS) if os.path.isdir(os.path.join(CARPETA_DATOS, d)) and d.startswith("vid_")]
    
    for sub in subfolders:
        sub_path = os.path.join(CARPETA_DATOS, sub)
        for file_name in os.listdir(sub_path):
            if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(file_name)[0]
                txt_name = name + ".txt"
                txt_path = os.path.join(sub_path, txt_name)
                
                # Ignorar classes.txt y verificar contenido en etiquetas
                if os.path.exists(txt_path) and txt_name != "classes.txt":
                    with open(txt_path, 'r') as f:
                        content = f.read().strip()
                    if content:
                        labeled_pairs.append((os.path.join(sub_path, file_name), txt_path))
                        
    print(f"  ✓ Imágenes anotadas totales encontradas: {len(labeled_pairs)}")
    
    if len(labeled_pairs) < 5:
        print("[!] Error: No hay suficientes datos anotados para entrenar.")
        return False

    # Mezclar y dividir en train/val
    random.seed(42)
    random.shuffle(labeled_pairs)
    
    split_idx = int(len(labeled_pairs) * (1.0 - VAL_SPLIT))
    train_pairs = labeled_pairs[:split_idx]
    val_pairs = labeled_pairs[split_idx:]
    
    print(f"  ✓ Imágenes para entrenamiento (train): {len(train_pairs)}")
    print(f"  ✓ Imágenes para validación (val): {len(val_pairs)}")
    
    # Copiar archivos asignándoles un prefijo único de su carpeta de origen
    def copiar_datos(pairs, subset):
        for img_path, txt_path in pairs:
            folder_orig = os.path.basename(os.path.dirname(img_path))
            filename = os.path.basename(img_path)
            name_only, ext = os.path.splitext(filename)
            
            new_img_name = f"{folder_orig}_{name_only}{ext}"
            new_txt_name = f"{folder_orig}_{name_only}.txt"
            
            shutil.copy2(img_path, os.path.join(DATASET_DIR, "images", subset, new_img_name))
            shutil.copy2(txt_path, os.path.join(DATASET_DIR, "labels", subset, new_txt_name))
            
    copiar_datos(train_pairs, "train")
    copiar_datos(val_pairs, "val")
    
    # Crear data.yaml
    dataset_path_formatted = DATASET_DIR.replace('\\', '/')
    yaml_content = f"""path: {dataset_path_formatted}
train: images/train
val: images/val

nc: 1
names: ['Humedad']
"""
    with open(YAML_PATH, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
        
    print(f"  ✓ Configuración creada en: {YAML_PATH}")
    return True


def entrenar_modelo():
    """
    Entrena el modelo YOLOv8 usando CUDA/GPU si está disponible.
    """
    print("\n" + "=" * 60)
    print(" PASO 2: Entrenando modelo YOLOv8")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  - Ejecutando en: {device.upper()}")
    
    model = YOLO(MODELO_BASE)
    
    model.train(
        data=YAML_PATH,
        epochs=EPOCAS,         # Usa las 150 épocas configuradas arriba
        batch=LOTE,            # Usa batch 16
        imgsz=IMG_SIZE,
        lr0=LR,
        device=device,
        project=r"C:\Users\Mariana\Downloads\modelo-humedad-fisuras-main\modelo-humedad-fisuras-main\runs",
        name=NOMBRE_PROYECTO,
        workers=0,
        exist_ok=True,
        patience=30,           # Detener si no mejora en 30 épocas para evitar sobreajuste
        # Aumentos de datos para combatir la confusión con sombras y orientaciones del dron:
        flipud=0.5,            # Volteo vertical (esencial para tomas aéreas/techos)
        hsv_v=0.6,             # Aumento de brillo/contraste (obliga al modelo a ignorar sombras puras)
        degrees=10.0,          # Rotaciones leves del dron
        mixup=0.05             # Mezcla de imágenes para robustez frente al ruido de fondo
    )
    
    print("  ✓ Entrenamiento finalizado con éxito.")


def evaluar_modelo():
    """
    Evalúa el rendimiento final y muestra métricas de validación.
    """
    print("\n" + "=" * 60)
    print(" PASO 3: Ejecutando validación de métricas")
    print("=" * 60)
    
    best_model_path = rf"C:\Users\Mariana\Downloads\modelo-humedad-fisuras-main\modelo-humedad-fisuras-main\runs\{NOMBRE_PROYECTO}\weights\best.pt"
    
    if not os.path.exists(best_model_path):
        print(f"[!] No se encontraron pesos guardados en: {best_model_path}")
        return
        
    model = YOLO(best_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    metricas = model.val(
        data=YAML_PATH,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False
    )
    
    print("\n" + "=" * 60)
    print(" METRICAS DE EVALUACION FINAL:")
    print("=" * 60)
    print(f"  - Precisión (Precision):  {metricas.box.mp:.4f}")
    print(f"  - Sensibilidad (Recall):   {metricas.box.mr:.4f}")
    print(f"  - mAP50:                   {metricas.box.map50:.4f}")
    print(f"  - mAP50-95:                {metricas.box.map:.4f}")


if __name__ == '__main__':
    try:
        if organizar_dataset():
            entrenar_modelo()
            evaluar_modelo()
            
            # Mover resultados de la carpeta temporal a la carpeta del proyecto
            print("\nMoviendo resultados finales de entrenamiento al espacio de trabajo...")
            src_run = rf"C:\Users\Mariana\Downloads\modelo-humedad-fisuras-main\modelo-humedad-fisuras-main\runs\{NOMBRE_PROYECTO}"
            dst_run = os.path.join(BASE_DIR, "Detección y Clasificación de Humedad Activa y Restos de Humedad", "runs", NOMBRE_PROYECTO)
            
            os.makedirs(os.path.dirname(dst_run), exist_ok=True)
            if os.path.exists(src_run):
                if os.path.exists(dst_run):
                    shutil.rmtree(dst_run)
                shutil.move(src_run, dst_run)
                print(f"  ✓ Resultados guardados en: {dst_run}")
            
            # Limpieza final de temporales
            print("\nLimpiando archivos temporales...")
            if os.path.exists(DATASET_DIR):
                shutil.rmtree(DATASET_DIR)
            if os.path.exists(YAML_PATH):
                os.remove(YAML_PATH)
                
            print("\n  ✓ ¡PROCESO DE ENTRENAMIENTO COMPLETO!")
            
    except KeyboardInterrupt:
        print("\n[!] Proceso interrumpido.")
    except Exception as e:
        print(f"\n[!] Ocurrió un error: {e}")
        import traceback
        traceback.print_exc()
        
    input("\nPresiona Enter para salir...")

