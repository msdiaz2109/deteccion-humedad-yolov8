import os
import sys
import time

try:
    import cv2
except ImportError:
    print("\n[!] Error: No se encontró la librería 'opencv-python' (cv2).")
    print("Para instalarla, ejecuta el siguiente comando en tu terminal:")
    print("    pip install opencv-python\n")
    input("Presiona Enter para salir...")
    sys.exit(1)

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_short_path(long_path):
    """
    Obtiene la ruta corta (8.3) de Windows para evitar problemas con Unicode/acentos en OpenCV.
    """
    if os.name != 'nt':
        return long_path
    try:
        import ctypes
        from ctypes import wintypes
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        GetShortPathNameW.restype = wintypes.DWORD
        
        buf_size = GetShortPathNameW(long_path, None, 0)
        if buf_size == 0:
            return long_path
        buf = ctypes.create_unicode_buffer(buf_size)
        if GetShortPathNameW(long_path, buf, buf_size) > 0:
            return buf.value
    except Exception:
        pass
    return long_path

def get_video_path():
    # Si se pasa como argumento de línea de comandos, lo usamos
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            return path
        else:
            print(f"[!] El archivo especificado en los argumentos no existe: {path}")

    # Si no hay argumentos, solicitamos interactividad
    while True:
        clear_screen()
        print("=" * 60)
        print("          EXTRACTOR DE IMÁGENES DE VIDEO (60 FPS / TODO)")
        print("=" * 60)
        print("Arrastra el archivo de video a esta ventana y presiona Enter,")
        print("o escribe la ruta completa del archivo de video.")
        print("-" * 60)
        
        path = input("Ruta del video: ").strip()
        
        # Eliminar prefijos de PowerShell como '& ' que se agregan al arrastrar archivos
        if path.startswith('&'):
            path = path[1:].strip()
            
        # Windows suele añadir comillas al arrastrar y soltar
        path = path.strip('\'"')
        
        if not path:
            continue
            
        if os.path.exists(path) and os.path.isfile(path):
            return path
        else:
            print(f"\n[!] Error: El archivo no existe o la ruta es incorrecta: '{path}'")
            input("Presiona Enter para intentar de nuevo...")

def extract_frames(video_path, image_format="jpg", target_fps=6):
    # Usar ruta corta de Windows para evitar problemas con Unicode/acentos en cv2.VideoCapture
    short_video_path = get_short_path(video_path)
    # Cargar el video
    cap = cv2.VideoCapture(short_video_path)
    if not cap.isOpened():
        print(f"\n[!] Error: No se pudo abrir el video '{video_path}'")
        return False

    # Obtener detalles del video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Obtener el nombre base del video para crear la carpeta
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    # Crear la carpeta de salida directamente en Imagenes del proyecto (sin el sufijo _imagenes)
    output_dir = os.path.join(
        r"C:\Users\Mariana\Desktop\Delfín\Detección y Clasificación de Humedad Activa y Restos de Humedad\Imagenes",
        video_basename
    )
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determinar si filtramos por FPS
    use_fps_filter = fps > target_fps
    estimated_frames = int(total_frames * (target_fps / fps)) if use_fps_filter else total_frames

    print("\n" + "=" * 60)
    print(f"Información del Video:")
    print(f"  - Nombre: {os.path.basename(video_path)}")
    print(f"  - Resolución: {width}x{height}")
    print(f"  - Fotogramas por segundo (FPS) original: {fps:.2f}")
    if use_fps_filter:
        print(f"  - FPS de extracción: {target_fps}")
        print(f"  - Total estimado de fotogramas a extraer: {estimated_frames}")
    else:
        print(f"  - Total de fotogramas a extraer: {total_frames}")
    print(f"  - Carpeta de salida: {output_dir}")
    print("=" * 60)

    # Determinar el padding para que las imágenes se ordenen correctamente (ej. 0001, 0002)
    padding = max(4, len(str(estimated_frames)))

    print("\nIniciando extracción. Por favor, espera...\n")
    start_time = time.time()
    
    frame_index = -1
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_index += 1  # 0-indexed
        
        if use_fps_filter:
            target_index = int(saved_count * (fps / target_fps))
            if frame_index < target_index:
                continue
        
        saved_count += 1
        
        # Determinar prefijo del archivo de acuerdo al formato solicitado (vidx_yyyy)
        prefix = ""
        if video_basename.lower().startswith("vid_"):
            parts = video_basename.split("_")
            try:
                x = int(parts[1])
                prefix = f"vid{x}_"
            except (IndexError, ValueError):
                prefix = f"{video_basename}_"
        else:
            prefix = f"{video_basename}_"

        # Generar nombre de archivo (ej: vid11_0001.jpg)
        filename = os.path.join(output_dir, f"{prefix}{saved_count:04d}.{image_format}")
        
        # Guardar imagen (método seguro para rutas con acentos/Unicode en Windows)
        try:
            ext = f".{image_format}"
            is_success, img_encoded = cv2.imencode(ext, frame)
            if is_success:
                with open(filename, "wb") as f:
                    f.write(img_encoded.tobytes())
            else:
                cv2.imwrite(filename, frame)
        except Exception:
            # Fallback en caso de cualquier error con open/write
            cv2.imwrite(filename, frame)

        # Mostrar progreso
        if total_frames > 0:
            percent = (frame_index + 1) / total_frames * 100
            bar_length = 30
            filled_length = int(round(bar_length * (frame_index + 1) / total_frames))
            bar = '#' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f"\rProgreso: |{bar}| {frame_index + 1}/{total_frames} ({percent:.1f}%) - Guardados: {saved_count}")
        else:
            sys.stdout.write(f"\rFotogramas leídos: {frame_index + 1} - Guardados: {saved_count}")
        sys.stdout.flush()

    cap.release()
    end_time = time.time()
    duration = end_time - start_time

    print(f"\n\n[OK] ¡Extracción completada con éxito!")
    print(f"  - Total guardado: {saved_count} imágenes")
    print(f"  - Tiempo transcurrido: {duration:.2f} segundos")
    print(f"  - Carpeta: {output_dir}\n")
    return True

if __name__ == "__main__":
    try:
        video_path = get_video_path()
        
        # Preguntar el formato de imagen preferido
        print("\nElige el formato de las imágenes:")
        print("  1) JPG (Recomendado: menor tamaño de archivo, procesamiento rápido)")
        print("  2) PNG (Calidad perfecta sin pérdida, pero archivos muy grandes)")
        choice = input("Selecciona una opción (1 o 2) [Por defecto: 1]: ").strip()
        
        img_format = "png" if choice == "2" else "jpg"
        
        # Preguntar la tasa de fotogramas objetivo
        print("\nConfiguración de extracción:")
        print("  - El video original puede tener 30 FPS o más.")
        fps_choice = input("¿Cuántas imágenes por segundo deseas extraer? [Por defecto: 6]: ").strip()
        try:
            target_fps = int(fps_choice) if fps_choice else 6
        except ValueError:
            target_fps = 6
            print("[i] Entrada no válida. Se usará el valor por defecto: 6 FPS")
            
        success = extract_frames(video_path, image_format=img_format, target_fps=target_fps)
        input("Presiona Enter para salir...")
    except KeyboardInterrupt:
        print("\n\n[!] Operación cancelada por el usuario.")
    except Exception as e:
        print(f"\n\n[!] Ocurrió un error inesperado: {e}")
        input("Presiona Enter para salir...")
