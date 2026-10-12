"""
================================================================================
SCRIPT 2: PRUEBA INTERACTIVA DE MODELO EN VIDEOS
================================================================================
Este script:
1. Muestra un menu para seleccionar que modelo/carpeta de entrenamiento usar (entrenamiento1, entrenamiento2, etc.).
2. Muestra un menu para seleccionar que video probar de la carpeta 'Videos/'.
3. Realiza la inferencia trazando:
   - Humedad Activa (Clase 0) en AZUL (Umbral de confianza >= 0.20)
   - Humedad Pasiva / Restos de Humedad (Clase 1) en ROJO (Umbral sensible >= 0.12)
4. Guarda el video resultante en la misma carpeta del modelo seleccionado (runs/entrenamientoX/resultados/).
"""

import os
import sys
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "Detección y Clasificación de Humedad Activa y Restos de Humedad", "runs")
VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUTS_DIR = os.path.join(BASE_DIR, "resultados")


def buscar_modelos_entrenados():
    modelos = []
    
    # 1. Modelo principal incluido en weights/
    best_empaquetado = os.path.join(WEIGHTS_DIR, "best.pt")
    if os.path.exists(best_empaquetado):
        modelos.append(("Modelo Principal Optimizado (weights/best.pt)", best_empaquetado, OUTPUTS_DIR))

    # 2. Modelos adicionales generados en runs/
    if os.path.exists(RUNS_DIR):
        for item in sorted(os.listdir(RUNS_DIR)):
            item_path = os.path.join(RUNS_DIR, item)
            best_pt = os.path.join(item_path, "weights", "best.pt")
            if os.path.isdir(item_path) and os.path.exists(best_pt):
                modelos.append((f"{item} (runs/{item})", best_pt, item_path))

    return modelos


def seleccionar_modelo(lista_modelos):
    print("=" * 70)
    print(" 1. SELECCION DE MODELO DE ENTRENAMIENTO")
    print("=" * 70)

    for i, (nombre, pt_path, _) in enumerate(lista_modelos, 1):
        print(f"  [{i:2d}] {nombre}")

    print("=" * 70)

    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        idx = int(sys.argv[1]) - 1
        if 0 <= idx < len(lista_modelos):
            print(f" -> Modelo seleccionado: {lista_modelos[idx][0]}")
            return lista_modelos[idx]

    while True:
        try:
            opcion = input(f"Ingresa el numero de modelo a usar (1 - {len(lista_modelos)}): ").strip()
            if opcion.isdigit():
                idx = int(opcion) - 1
                if 0 <= idx < len(lista_modelos):
                    return lista_modelos[idx]
            print(f"[!] Por favor ingresa un numero valido entre 1 y {len(lista_modelos)}.")
        except (KeyboardInterrupt, EOFError):
            print("\n[!] Operacion cancelada por el usuario.")
            sys.exit(0)


def buscar_videos(directorio):
    extensiones = ('.mp4', '.avi', '.mov', '.mkv')
    lista_videos = []

    for root, dirs, files in os.walk(directorio):
        if os.path.basename(root).lower() == 'resultados':
            continue
        for f in files:
            if f.lower().endswith(extensiones):
                lista_videos.append(os.path.join(root, f))

    return sorted(lista_videos)


def seleccionar_video(lista_videos):
    print("\n" + "=" * 70)
    print(" 2. SELECCION DE VIDEO PARA ANALISIS")
    print("=" * 70)

    for i, vid_path in enumerate(lista_videos, 1):
        rel_path = os.path.relpath(vid_path, VIDEOS_DIR)
        tamano_mb = os.path.getsize(vid_path) / (1024 * 1024)
        print(f"  [{i:2d}] {rel_path} ({tamano_mb:.1f} MB)")

    print("=" * 70)

    if len(sys.argv) > 2 and sys.argv[2].isdigit():
        idx = int(sys.argv[2]) - 1
        if 0 <= idx < len(lista_videos):
            print(f" -> Video seleccionado: {os.path.basename(lista_videos[idx])}")
            return lista_videos[idx]

    while True:
        try:
            opcion = input(f"Ingresa el numero de video a analizar (1 - {len(lista_videos)}): ").strip()
            if opcion.isdigit():
                idx = int(opcion) - 1
                if 0 <= idx < len(lista_videos):
                    return lista_videos[idx]
            print(f"[!] Por favor ingresa un numero valido entre 1 y {len(lista_videos)}.")
        except (KeyboardInterrupt, EOFError):
            print("\n[!] Operacion cancelada por el usuario.")
            sys.exit(0)


def procesar_video(video_path, model, folder_destino):
    output_dir = os.path.join(folder_destino, "resultados")
    os.makedirs(output_dir, exist_ok=True)

    nombre_base = os.path.basename(video_path)
    stem, _ = os.path.splitext(nombre_base)
    salida_path = os.path.join(output_dir, f"resultado_{stem}.mp4")

    print(f"\nProcesando video: {nombre_base}")
    print(f"El video resultante se guardara en:\n  {salida_path}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] Error: No se pudo abrir el archivo de video: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(salida_path, fourcc, fps, (width, height))

    frame_count = 0
    conf_activa = 0.20
    conf_pasiva = 0.12

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Correr modelo a un umbral bajo para capturar la humedad pasiva tenue
        results = model(frame, conf=0.10, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                conf_score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                if cls_idx == 0 and conf_score >= conf_activa:
                    label = f"Humedad Activa {conf_score:.2f}"
                    color = (255, 0, 0)  # Azul (BGR)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                elif cls_idx == 1 and conf_score >= conf_pasiva:
                    label = f"Humedad Pasiva {conf_score:.2f}"
                    color = (0, 0, 255)  # Rojo (BGR)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)

        if total_frames > 0 and frame_count % 100 == 0:
            progreso = (frame_count / total_frames) * 100
            print(f"  Procesados {frame_count}/{total_frames} fotogramas ({progreso:.1f}%)...")

    cap.release()
    out.release()

    print(f"\n[OK] Procesamiento completado con exito!")
    print(f"Video resultante guardado en:\n  {salida_path}")


def main():
    modelos = buscar_modelos_entrenados()
    if not modelos:
        print("[!] Error: No se encontraron modelos entrenados en 'weights/' ni en 'runs/'.")
        print("Por favor ejecuta primero 'python 1_entrenar.py' o coloca 'best.pt' en la carpeta 'weights/'.")
        sys.exit(1)

    videos = buscar_videos(VIDEOS_DIR) if os.path.exists(VIDEOS_DIR) else []
    if not videos:
        print(f"[!] No se encontraron videos de prueba en la carpeta: {VIDEOS_DIR}")
        print("Puedes ingresar la ruta directa de un video (.mp4) para analizar:")
        ruta_directa = input("Ruta del archivo de video: ").strip().strip('\'"')
        if ruta_directa and os.path.isfile(ruta_directa):
            videos = [ruta_directa]
        else:
            print("[!] Archivo no encontrado. Saliendo.")
            sys.exit(1)

    nombre_modelo, best_pt, folder_modelo = seleccionar_modelo(modelos)
    video_seleccionado = seleccionar_video(videos) if len(videos) > 1 else videos[0]

    print(f"\nCargando modelo '{nombre_modelo}' ({best_pt})...")
    model = YOLO(best_pt)

    procesar_video(video_seleccionado, model, folder_modelo)


if __name__ == '__main__':
    main()
