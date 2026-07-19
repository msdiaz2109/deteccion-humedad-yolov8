import os
import cv2
from ultralytics import YOLO

# ============================================================
#  CONFIGURACIÓN DE LA EVALUACIÓN EN VIDEO
# ============================================================

# Ruta base del proyecto
BASE_DIR = r"c:\Users\Mariana\Desktop\Delfín"

# Nombre del proyecto donde se guardaron los pesos de tu entrenamiento
NOMBRE_PROYECTO = "humedad_detector_global"

# Ruta del mejor modelo guardado localmente en tu carpeta de proyecto
MODEL_PATH = os.path.join(BASE_DIR, "Detección y Clasificación de Humedad Activa y Restos de Humedad", "runs", NOMBRE_PROYECTO, "weights", "best.pt")

# Ruta del video de entrada y salida
VIDEO_INPUT = r"C:\Users\Mariana\Downloads\modelo-humedad-fisuras-main\modelo-humedad-fisuras-main\data\raw\videos\vid11.mp4"
VIDEO_OUTPUT = os.path.join(BASE_DIR, "Detección y Clasificación de Humedad Activa y Restos de Humedad", "runs", NOMBRE_PROYECTO, "video11_detecciones.mp4")


# Resolución de entrada de la imagen para la predicción
IMG_SIZE = 640

# Confianza mínima para dibujar la detección
# NOTA: 0.15 era demasiado bajo y causaba detecciones duplicadas.
# 0.35 filtra las cajas de baja confianza que se solapaban.
CONF_THRESHOLD = 0.35

# Umbral de IoU para NMS (Non-Maximum Suppression)
# Cuanto MÁS BAJO, más agresivamente fusiona cajas solapadas.
# 0.4 significa: si dos cajas se solapan más del 40%, eliminar la de menor confianza.
IOU_THRESHOLD = 0.4

# Máximo de detecciones por frame
MAX_DET = 50


def evaluar_en_video():
    """
    Carga el modelo entrenado y procesa el video frame a frame,
    dibujando las cajas de detección sobre las humedades.
    """
    print("=" * 60)
    print(" EVALUACIÓN DEL MODELO EN VIDEO 11")
    print("=" * 60)
    
    # Verificar existencia del modelo
    if not os.path.exists(MODEL_PATH):
        print(f"\n[!] Error: No se encontró el modelo entrenado en {MODEL_PATH}")
        print("    Asegúrate de ejecutar primero el script 'entrenar_global.py' con éxito.")
        return
        
    # Verificar existencia del video
    if not os.path.exists(VIDEO_INPUT):
        print(f"\n[!] Error: No se encontró el video de entrada en: {VIDEO_INPUT}")
        return
        
    print(f"\n  - Cargando modelo: {MODEL_PATH}")
    print(f"  - Video origen: {VIDEO_INPUT}")
    print(f"  - Guardando resultado en: {VIDEO_OUTPUT}")
    
    # Cargar modelo entrenado
    model = YOLO(MODEL_PATH)
    
    # Abrir video
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print("[!] Error al abrir el video de entrada.")
        return
        
    # Obtener propiedades físicas del video de entrada
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  - FPS: {fps} | Resolución: {width}x{height} | Total de frames: {total_frames}")
    
    # Asegurar que existe la carpeta del archivo de salida
    os.makedirs(os.path.dirname(VIDEO_OUTPUT), exist_ok=True)
    
    # Configurar VideoWriter para exportar a MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))
    
    # Colores BGR (Verde para Humedad [clase 0], Rojo para Fisura [clase 1])
    COLORES = [(0, 255, 0), (0, 0, 255)]
    frame_idx = 0
    
    print("\n  Procesando frames del video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # Predicción en el frame actual con NMS ajustado para evitar cajas duplicadas
        results = model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,       # Fusiona cajas solapadas >40%
            agnostic_nms=True,       # Fusiona cajas sin importar la clase
            max_det=MAX_DET,         # Limita detecciones por frame
            classes=[0],             # Solo detectar Humedad (clase 0), ignorar Fisura
            verbose=False
        )
        
        # Dibujar las detecciones
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    
                    color = COLORES[cls_id % len(COLORES)]
                    
                    # Dibujar caja contenedora
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Etiqueta de texto (Clase + Confianza)
                    lbl = f"{cls_name} {conf:.2f}"
                    txt_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    
                    # Fondo y texto de la etiqueta
                    cv2.rectangle(frame, (x1, y1 - txt_size[1] - 8), (x1 + txt_size[0] + 4, y1), color, -1)
                    cv2.putText(frame, lbl, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        # Escribir frame resultante en el video
        out.write(frame)
        
        # Mostrar barra de progreso simple en consola
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames) * 100
            print(f"    Frame {frame_idx}/{total_frames} ({pct:.1f}%) completado.")
            
    # Liberar recursos de lectura y escritura
    cap.release()
    out.release()
    
    print("\n" + "=" * 60)
    print("  ✓ ¡PROCESAMIENTO DE VIDEO FINALIZADO!")
    print("=" * 60)
    print(f"  ✓ Archivo guardado exitosamente en: {VIDEO_OUTPUT}")


if __name__ == '__main__':
    try:
        evaluar_en_video()
    except KeyboardInterrupt:
        print("\n[!] Proceso de video cancelado.")
    except Exception as e:
        print(f"\n[!] Ocurrió un error inesperado: {e}")
        
    input("\nPresiona Enter para finalizar...")
