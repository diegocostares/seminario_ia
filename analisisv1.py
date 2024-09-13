import os
import time
import signal
import csv
from datetime import datetime
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2

# Variables globales
CAPTURE_INTERVAL = 20  # Segundos entre capturas de pantalla (por defecto 20 segundos)
ENABLE_VISUALIZATION = False  # Controla si se muestra la ventana de video (True/False)
DETECTION_LIST = ["person"]  # Siempre detecta personas
BLACKLIST = ["cell phone"]  # Nunca detecta teléfonos

# Crear la carpeta de salida si no existe
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inicializar archivo CSV para almacenar los resultados
csv_filename = "detections.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Frame", "Fecha", "Hora", "Objetos Detectados", "Confidencias", "Coordenadas", "Tamaño del área"])

# Inicializar la cámara con resolución ajustada
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
picam2.configure(config)
picam2.start()

# Cargar el modelo YOLOv8 preentrenado
model = YOLO("yolov8x.pt")

frame_count = 0  # Contador de frames

# Manejador para detener la ejecución con Ctrl+C
def signal_handler(sig, frame):
    print("Deteniendo captura...")
    picam2.stop()
    cv2.destroyAllWindows()
    exit(0)

# Asignar el manejador de la señal para Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Función para filtrar detecciones según las listas de inclusión y exclusión
def filtrar_detecciones(results):
    detected_objects = []
    confidences = []
    coordinates = []
    areas = []
    for result in results[0].boxes.data:
        object_class = int(result[5])
        confidence = float(result[4])
        object_name = model.names[object_class]

        # Aplicar lista blanca (detección) y lista negra (exclusión)
        if object_name in BLACKLIST:
            continue
        if object_name in DETECTION_LIST or "person" in object_name:
            detected_objects.append(object_name)
            confidences.append(round(confidence, 2))  # Redondear a 2 decimales
            
            # Coordenadas de la caja delimitadora
            x_min, y_min, x_max, y_max = map(int, result[:4])
            coordinates.append([x_min, y_min, x_max, y_max])

            # Calcular el área de la caja
            width = x_max - x_min
            height = y_max - y_min
            areas.append((width, height))

    return detected_objects, confidences, coordinates, areas

# Función para realizar la captura
def captura_deteccion():
    global frame_count
    # Capturar un frame de la cámara
    frame = picam2.capture_array()

    # Convertir a 3 canales (de 4 a 3, eliminando el canal alfa si está presente)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Realizar la detección en el frame actual
    results = model(frame)

    # Filtrar detecciones según las listas de inclusión y exclusión
    detected_objects, confidences, coordinates, areas = filtrar_detecciones(results)

    # Obtener la fecha y hora actuales
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # Guardar la información en el archivo CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([frame_count, current_date, current_time, detected_objects, confidences, coordinates, areas])

    # Guardar el frame anotado
    annotated_frame = results[0].plot()
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, annotated_frame)

    # Mostrar el frame con las detecciones (si la visualización está habilitada)
    if ENABLE_VISUALIZATION:
        cv2.imshow("Detección en Tiempo Real", annotated_frame)
        cv2.waitKey(1)  # Necesario para que OpenCV actualice la ventana

    frame_count += 1

# Iniciar la captura de video en tiempo real
try:
    while True:
        start_time = time.time()
        captura_deteccion()
        elapsed_time = time.time() - start_time
        time.sleep(max(0, CAPTURE_INTERVAL - elapsed_time))
finally:
    picam2.stop()
    cv2.destroyAllWindows()
