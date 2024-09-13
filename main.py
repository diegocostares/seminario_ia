import os
import time
import signal
import csv
import logging
from datetime import datetime
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import numpy as np
from multiprocessing import Process, Queue

# Configuración del registro (logging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Clase de Configuración
class Config:
    ENABLE_VISUALIZATION = False  # Controla si se muestra la ventana de video (True/False)
    DETECTION_LIST = ["person"]  # Lista de objetos a detectar
    CONFIDENCE_THRESHOLD = 0.5  # Confianza mínima para detectar objetos
    OUTPUT_FOLDER = "output"  # Carpeta de salida para imágenes y CSV
    CSV_FILENAME = "detections.csv"  # Nombre del archivo CSV para almacenar resultados
    MIN_MOTION_AREA = 5000  # Área mínima de movimiento para considerar que hay cambio
    RESIZE_WIDTH = 640  # Ancho al que se redimensionará la imagen para detección de movimiento
    CAMERA_RESOLUTION = (640, 360)  # Resolución de la cámara (ancho, alto)
    MODEL_NAME = "yolov8x.pt"  # Modelo YOLOv8 a utilizar (nano para mayor velocidad)

# Clase para manejar la detección de objetos
class ObjectDetector:
    def __init__(self, config):
        self.config = config
        # Cargar el modelo YOLOv8 especificado
        self.model = YOLO(self.config.MODEL_NAME)
        self.classes_ids = self.get_classes_ids()
        logging.info(f"Modelo YOLO '{self.config.MODEL_NAME}' cargado y listo para usar.")

    def get_classes_ids(self):
        if not self.config.DETECTION_LIST:
            return None  # Detectar todas las clases si la lista está vacía
        classes_ids = []
        for object_name in self.config.DETECTION_LIST:
            if object_name in self.model.names.values():
                class_id = list(self.model.names.values()).index(object_name)
                classes_ids.append(class_id)
            else:
                logging.warning(f"Objeto '{object_name}' no encontrado en las clases del modelo.")
        return classes_ids

    def detect_objects(self, frame):
        # Realizar la detección en el frame actual
        results = self.model(frame, classes=self.classes_ids, conf=self.config.CONFIDENCE_THRESHOLD)
        return results

    def filter_detections(self, results):
        detected_objects = []
        confidences = []
        coordinates = []
        areas = []
        # Verificar si hay detecciones
        if results[0].boxes is not None:
            for result in results[0].boxes.data:
                object_class = int(result[5])
                confidence = float(result[4])
                object_name = self.model.names[object_class]

                # Filtrar según confianza mínima
                if confidence < self.config.CONFIDENCE_THRESHOLD:
                    continue
                detected_objects.append(object_name)
                confidences.append(round(confidence, 2))  # Redondear a 2 decimales

                # Coordenadas de la caja delimitadora
                x_min, y_min, x_max, y_max = map(int, result[:4])
                coordinates.append([x_min, y_min, x_max, y_max])

                # Calcular el área de la caja
                width = x_max - x_min
                height = y_max - y_min
                areas.append((width, height))
        else:
            logging.info("No se detectaron objetos en este frame.")

        return detected_objects, confidences, coordinates, areas

# Clase para manejar la cámara y las capturas
class CameraHandler:
    def __init__(self, config, frame_queue):
        self.config = config
        self.frame_queue = frame_queue
        self.picam2 = Picamera2()
        self.configure_camera()
        logging.info("Cámara inicializada y configurada.")

    def configure_camera(self):
        # Configurar la cámara con la resolución especificada
        config = self.picam2.create_preview_configuration(main={"size": self.config.CAMERA_RESOLUTION})
        self.picam2.configure(config)
        self.picam2.start()

    def capture_frames(self):
        prev_frame = None
        while True:
            # Capturar un frame de la cámara
            frame = self.picam2.capture_array()

            # Convertir el frame a escala de grises para detección de movimiento
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.config.RESIZE_WIDTH, int(gray.shape[0] * self.config.RESIZE_WIDTH / gray.shape[1])))
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is None:
                prev_frame = gray
                continue

            # Calcular la diferencia entre el frame actual y el anterior
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < self.config.MIN_MOTION_AREA:
                    continue
                motion_detected = True
                break

            # Si se detecta movimiento, enviar el frame para procesamiento
            if motion_detected:
                logging.info("Movimiento detectado.")
                self.frame_queue.put(frame)

            # Actualizar el frame anterior
            prev_frame = gray

            # Controlar la velocidad de captura
            time.sleep(0.1)  # Pequeña pausa para evitar sobrecarga de CPU

    def release(self):
        self.picam2.stop()
        logging.info("Cámara liberada.")

# Clase principal que coordina el proceso
class DetectionApp:
    def __init__(self):
        self.config = Config()
        self.frame_queue = Queue()
        self.detector_process = None
        self.camera_process = None
        self.frame_count = 0

        # Crear carpeta de salida si no existe
        if not os.path.exists(self.config.OUTPUT_FOLDER):
            os.makedirs(self.config.OUTPUT_FOLDER)
            logging.info(f"Carpeta de salida creada: {self.config.OUTPUT_FOLDER}")

        # Inicializar el archivo CSV
        self.init_csv()

    def init_csv(self):
        self.csv_filepath = os.path.join(self.config.OUTPUT_FOLDER, self.config.CSV_FILENAME)
        with open(self.csv_filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Frame", "Fecha", "Hora", "Objetos Detectados", "Confidencias", "Coordenadas", "Tamaño del área"])
        logging.info(f"Archivo CSV inicializado: {self.csv_filepath}")

    def start(self):
        # Iniciar procesos de captura y procesamiento
        self.detector_process = Process(target=self.process_frames)
        self.camera_process = Process(target=self.capture_frames)

        self.detector_process.start()
        self.camera_process.start()

    def capture_frames(self):
        camera_handler = CameraHandler(self.config, self.frame_queue)
        try:
            camera_handler.capture_frames()
        except KeyboardInterrupt:
            pass
        finally:
            camera_handler.release()

    def process_frames(self):
        detector = ObjectDetector(self.config)
        try:
            while True:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()

                    # Verificar si el frame tiene 4 canales (por ejemplo, RGBA) y convertir a BGR
                    if frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                    # Realizar detección
                    results = detector.detect_objects(frame)

                    # Filtrar detecciones
                    detected_objects, confidences, coordinates, areas = detector.filter_detections(results)

                    # Anotar frame con detecciones
                    annotated_frame = results[0].plot()

                    # Corregir el espacio de color antes de guardar (de RGB a BGR)
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

                    # Guardar resultados
                    self.save_results(frame, annotated_frame, detected_objects, confidences, coordinates, areas)

                    # Mostrar visualización si está habilitada
                    if self.config.ENABLE_VISUALIZATION:
                        cv2.imshow("Detección en Tiempo Real", annotated_frame)
                        cv2.waitKey(1)

                    self.frame_count += 1
                else:
                    time.sleep(0.1)  # Esperar un poco si no hay frames en la cola
        except KeyboardInterrupt:
            pass
        finally:
            if self.config.ENABLE_VISUALIZATION:
                cv2.destroyAllWindows()
            logging.info("Proceso de detección finalizado.")

    def save_results(self, frame, annotated_frame, detected_objects, confidences, coordinates, areas):
        # Guardar la imagen anotada con el nombre del frame actual
        frame_filename = os.path.join(self.config.OUTPUT_FOLDER, f"frame_{self.frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)

        # Obtener fecha y hora actuales
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # Guardar información en el archivo CSV
        with open(self.csv_filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                self.frame_count,
                current_date,
                current_time,
                detected_objects,
                confidences,
                coordinates,
                areas
            ])

    def signal_handler(self, sig, frame):
        logging.info("Deteniendo la aplicación...")
        self.detector_process.terminate()
        self.camera_process.terminate()
        self.detector_process.join()
        self.camera_process.join()
        exit(0)

    def run(self):
        # Manejar señal para Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        logging.info("Iniciando la aplicación de detección.")
        self.start()

        # Mantener el proceso principal vivo
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.signal_handler(None, None)

if __name__ == "__main__":
    app = DetectionApp()
    app.run()
