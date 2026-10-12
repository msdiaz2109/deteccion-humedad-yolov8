"""
================================================================================
SCRIPT 1: ENTRENAMIENTO DEL MODELO DE HUMEDAD ACTIVA Y HUMEDAD PASIVA
================================================================================
Este script:
1. Incluye todas las carpetas con anotaciones de ambas clases (incluyendo vid_012, vid_013, vid_017 y restos_de_humedad)
   para resolver el desbalance de datos y mejorar la deteccion de la Humedad Pasiva.
2. Mapea las 2 clases:
   - Clase 0: Humedad (Humedad Activa)
   - Clase 1: Restos de Humedad (Humedad Pasiva)
3. Incrementa la ganancia de clasificacion (cls=1.5) para forzar al modelo a priorizar
   los patrones de Humedad Pasiva.
4. Genera secuencialmente una nueva carpeta (entrenamiento1, entrenamiento2, etc.).
"""

import os
import sys
import shutil
import random
import yaml
import torch
from ultralytics import YOLO

import tempfile

# Configuracion Principal
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_IMAGENES = os.path.join(BASE_DIR, "Detección y Clasificación de Humedad Activa y Restos de Humedad", "Imagenes")
RUNS_DIR = os.path.join(BASE_DIR, "Detección y Clasificación de Humedad Activa y Restos de Humedad", "runs")

# Carpetas de entrenamiento optimizadas para cubrir todo el espectro de Humedad Pasiva
CARPETAS_ENTRENAMIENTO = [
    "vid_001", "vid_002", "vid_004", "vid_005", "vid_006", 
    "vid_011", "vid_012", "vid_013", "vid_014", "vid_015", 
    "vid_016", "vid_017", "restos_de_humedad"
]

# Directorio temporal de entrenamiento
DATASET_TEMP = os.path.join(tempfile.gettempdir(), "yolo_train_humedad_dual")
YAML_TEMP = os.path.join(DATASET_TEMP, "dataset.yaml")

MODELO_BASE = "yolov8n.pt"
EPOCAS = 50
BATCH_SIZE = 16
IMG_SIZE = 640
VAL_SPLIT = 0.20


def obtener_siguiente_nombre_entrenamiento():
    os.makedirs(RUNS_DIR, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(RUNS_DIR, f"entrenamiento{n}")):
        n += 1
    return f"entrenamiento{n}"


def preparar_dataset():
    print("=" * 70)
    print(" 1. Verificando y estructurando dataset balanceado de entrenamiento")
    print(f"    Carpetas: {CARPETAS_ENTRENAMIENTO}")
    print("=" * 70)

    if os.path.exists(DATASET_TEMP):
        shutil.rmtree(DATASET_TEMP)

    for subset in ["train", "val"]:
        os.makedirs(os.path.join(DATASET_TEMP, "images", subset), exist_ok=True)
        os.makedirs(os.path.join(DATASET_TEMP, "labels", subset), exist_ok=True)

    labeled_pairs = []
    total_c0, total_c1 = 0, 0

    for sub in CARPETAS_ENTRENAMIENTO:
        sub_path = os.path.join(CARPETA_IMAGENES, sub)
        if not os.path.exists(sub_path):
            continue

        c0_count, c1_count = 0, 0
        pairs_in_folder = 0

        for f in os.listdir(sub_path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                stem = os.path.splitext(f)[0]
                txt_name = stem + ".txt"
                txt_path = os.path.join(sub_path, txt_name)

                if os.path.exists(txt_path) and txt_name != "classes.txt":
                    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as tf:
                        lines = tf.readlines()

                    if lines:
                        pairs_in_folder += 1
                        labeled_pairs.append((os.path.join(sub_path, f), txt_path, sub))
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                cls_id = int(float(parts[0]))
                                if cls_id == 0:
                                    c0_count += 1
                                elif cls_id == 1:
                                    c1_count += 1

        total_c0 += c0_count
        total_c1 += c1_count
        print(f"  {sub:20s} | Imagenes: {pairs_in_folder:4d} | Humedad Activa (0): {c0_count:4d} | Humedad Pasiva (1): {c1_count:4d}")

    print("-" * 70)
    print(f"  [OK] Total de imagenes: {len(labeled_pairs)}")
    print(f"  [OK] Cajas Humedad Activa (0): {total_c0} | Cajas Humedad Pasiva (1): {total_c1}")

    if len(labeled_pairs) < 10:
        print("[!] Error: No hay imagenes suficientes para entrenar.")
        return False

    random.seed(42)
    random.shuffle(labeled_pairs)

    split_idx = int(len(labeled_pairs) * (1.0 - VAL_SPLIT))
    train_pairs = labeled_pairs[:split_idx]
    val_pairs = labeled_pairs[split_idx:]

    print(f"  [OK] Division: {len(train_pairs)} para Train, {len(val_pairs)} para Val.")

    def copiar_parejas(pairs, subset_name):
        for img_path, txt_path, folder_orig in pairs:
            filename = os.path.basename(img_path)
            stem, ext = os.path.splitext(filename)
            new_img_name = f"{folder_orig}_{stem}{ext}"
            new_txt_name = f"{folder_orig}_{stem}.txt"
            shutil.copy2(img_path, os.path.join(DATASET_TEMP, "images", subset_name, new_img_name))
            shutil.copy2(txt_path, os.path.join(DATASET_TEMP, "labels", subset_name, new_txt_name))

    copiar_parejas(train_pairs, "train")
    copiar_parejas(val_pairs, "val")

    path_formatted = DATASET_TEMP.replace('\\', '/')
    dataset_dict = {
        "path": path_formatted,
        "train": "images/train",
        "val": "images/val",
        "nc": 2,
        "names": ["Humedad", "Restos de Humedad"]
    }

    with open(YAML_TEMP, "w", encoding="utf-8") as f:
        yaml.dump(dataset_dict, f, allow_unicode=True, default_flow_style=False)

    print(f"  [OK] Configuracion creada en: {YAML_TEMP}")
    return True


def entrenar():
    nombre_ejecucion = obtener_siguiente_nombre_entrenamiento()

    print("\n" + "=" * 70)
    print(f" 2. Ejecutando Entrenamiento YOLOv8n Optimizado ({EPOCAS} Epocas)")
    print(f"    Nombre de corrida: {nombre_ejecucion}")
    print("=" * 70)

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Ejecutando en GPU: {gpu_name}")
    else:
        print("  Ejecutando en CPU para entrenamiento.")

    model = YOLO(MODELO_BASE)

    model.train(
        data=YAML_TEMP,
        epochs=EPOCAS,
        batch=BATCH_SIZE if device == 0 else 4,
        imgsz=IMG_SIZE,
        device=device,
        project=RUNS_DIR,
        name=nombre_ejecucion,
        exist_ok=True,
        workers=0,
        patience=25,
        cls=1.5,             # Aumento de ganancia para clase pasiva dificil
        box=8.5,             # Precision de cajas
        flipud=0.5,
        hsv_v=0.6,
        degrees=10.0,
        mixup=0.05,
        plots=True,
        verbose=True
    )

    run_folder = os.path.join(RUNS_DIR, nombre_ejecucion)
    pesos_finales = os.path.join(run_folder, "weights", "best.pt")
    
    print("\n" + "=" * 70)
    print(" 3. Validacion y Evaluacion del Modelo")
    print("=" * 70)

    if os.path.exists(pesos_finales):
        model_val = YOLO(pesos_finales)
        metricas = model_val.val(
            data=YAML_TEMP,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False
        )

        print("\n  METRICAS FINALES DE DESEMPEÑO:")
        print(f"  - Precision (Precision):  {metricas.box.mp:.4f}")
        print(f"  - Sensibilidad (Recall):   {metricas.box.mr:.4f}")
        print(f"  - mAP50:                   {metricas.box.map50:.4f}")
        print(f"  - mAP50-95:                {metricas.box.map:.4f}")

        print(f"\n  [OK] Entrenamiento guardado en:\n  {run_folder}")
    else:
        print(f"[!] No se encontraron los pesos en: {pesos_finales}")


def main():
    if preparar_dataset():
        entrenar()
        if os.path.exists(DATASET_TEMP):
            shutil.rmtree(DATASET_TEMP)


if __name__ == "__main__":
    main()
