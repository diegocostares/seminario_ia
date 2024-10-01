import csv
import logging
import os
import signal
import threading
from datetime import datetime
from queue import Empty, Queue

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# Configuración del registro (logging)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Config:
    ENABLE_VISUALIZATION = False     # Controla si se muestra la ventana de video (True/False)
    DETECTION_LIST = ["person"]      # Lista de objetos a detectar
    CONFIDENCE_THRESHOLD = 0.15      # Confianza mínima para detectar objetos
    OUTPUT_FOLDER = "output"         # Carpeta de salida para imágenes y CSV
    CSV_FILENAME = "detections.csv"  # Nombre del archivo CSV para almacenar resultados
    MIN_MOTION_AREA = 5000           # Área mínima de movimiento para considerar que hay cambio
    RESIZE_WIDTH = 640               # Ancho al que se redimensionará la imagen para detección de movimiento
    CAMERA_RESOLUTION = (640, 360)   # Resolución de la cámara (ancho, alto)
    MODEL_NAME = "yolov8x.pt"        # Modelo YOLOv8 a utilizar
    CSV_BUFFER_SIZE = 2              # Número de entradas antes de escribir en el CSV
    MOTION_DETECTION_COOLDOWN = 5    # Segundos de espera después de detectar movimiento
    BACKGROUND_SUBTRACTOR = {        # Configuración para createBackgroundSubtractorKNN
        "history": 500,
        "dist2Threshold": 400.0,
        "detectShadows": False,
    }


class ObjectDetector:
    """Clase para manejar la detección de objetos con YOLOv8"""

    def __init__(self, config):
        self.config = config
        # Cargar el modelo YOLOv8 especificado
        self.model = YOLO(self.config.MODEL_NAME)
        self.classes_ids = self.get_classes_ids()
        logging.info(f"Modelo YOLO '{self.config.MODEL_NAME}' cargado y listo para usar.")

    def get_classes_ids(self):
        if not self.config.DETECTION_LIST:
            return None
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
        if results[0].boxes is not None and len(results[0].boxes.data) > 0:
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


class FrameCapture(threading.Thread):
    """Hilo para capturar frames de la cámara y detectar movimiento"""

    def __init__(self, config, frame_queue, motion_event, stop_event):
        super().__init__()
        self.config = config
        self.frame_queue = frame_queue
        self.motion_event = motion_event
        self.stop_event = stop_event
        self.picam2 = Picamera2()
        self.configure_camera()
        # Configurar el sustractor de fondo KNN para detección de movimiento
        self.backSub = cv2.createBackgroundSubtractorKNN(
            history=self.config.BACKGROUND_SUBTRACTOR["history"],
            dist2Threshold=self.config.BACKGROUND_SUBTRACTOR["dist2Threshold"],
            detectShadows=self.config.BACKGROUND_SUBTRACTOR["detectShadows"],
        )
        logging.info("Cámara y sustractor de fondo inicializados y configurados.")

    def configure_camera(self):
        # Configurar la cámara con la resolución especificada
        config = self.picam2.create_preview_configuration(main={"size": self.config.CAMERA_RESOLUTION})
        self.picam2.configure(config)
        self.picam2.start()

    def run(self):
        last_motion_time = None

        while not self.stop_event.is_set():
            try:
                # Capturar un frame de la cámara
                frame = self.picam2.capture_array()

                # Aplicar el sustractor de fondo para detectar movimiento
                fg_mask = self.backSub.apply(frame)

                # Operaciones morfológicas para reducir el ruido
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                # Encontrar contornos
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                motion_detected = False
                for contour in contours:
                    if cv2.contourArea(contour) < self.config.MIN_MOTION_AREA:
                        continue
                    motion_detected = True
                    break

                current_time = datetime.now()

                if motion_detected:
                    if (
                        last_motion_time is None
                        or (current_time - last_motion_time).total_seconds() > self.config.MOTION_DETECTION_COOLDOWN
                    ):
                        logging.info("Movimiento detectado. Encolando frame para detección.")
                        self.frame_queue.put(frame)
                        last_motion_time = current_time
                        # Señalar que se ha detectado movimiento
                        self.motion_event.set()
                else:
                    # Si ha pasado el periodo de enfriamiento, resetear el evento
                    if (
                        last_motion_time
                        and (current_time - last_motion_time).total_seconds() > self.config.MOTION_DETECTION_COOLDOWN
                    ):
                        self.motion_event.clear()
                        last_motion_time = None

            except Exception as e:
                logging.error(f"Error en FrameCapture: {e}")

    def stop(self):
        self.stop_event.set()
        self.picam2.stop()
        logging.info("Cámara liberada y FrameCapture detenido.")


class DetectorThread(threading.Thread):
    """Hilo para la detección de objetos"""

    def __init__(self, config, frame_queue, motion_event, stop_event):
        super().__init__()
        self.config = config
        self.frame_queue = frame_queue
        self.motion_event = motion_event
        self.stop_event = stop_event
        self.frame_count = 0
        self.csv_buffer = []
        self.csv_filepath = os.path.join(self.config.OUTPUT_FOLDER, self.config.CSV_FILENAME)
        self.detector = ObjectDetector(self.config)

    def run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    # Esperar hasta que haya un frame disponible o se indique detenerse
                    frame = self.frame_queue.get(timeout=1)
                except Empty:
                    continue

                # Asegurar que el frame tiene 3 canales
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                # Realizar detección
                results = self.detector.detect_objects(frame)

                # Filtrar detecciones
                detected_objects, confidences, coordinates, areas = self.detector.filter_detections(results)

                # Anotar frame con detecciones
                annotated_frame = results[0].plot()

                # Guardar resultados
                self.save_results(annotated_frame, detected_objects, confidences, coordinates, areas)

                # Mostrar visualización si está habilitada
                if self.config.ENABLE_VISUALIZATION:
                    cv2.imshow("Detección en Tiempo Real", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_event.set()

                self.frame_count += 1

        except Exception as e:
            logging.error(f"Error en DetectorThread: {e}")
        finally:
            if self.config.ENABLE_VISUALIZATION:
                cv2.destroyAllWindows()
            # Escribir cualquier resultado pendiente en el CSV
            if self.csv_buffer:
                self.write_csv_buffer()
            logging.info("DetectorThread finalizado.")

    def save_results(self, annotated_frame, detected_objects, confidences, coordinates, areas):
        # Guardar la imagen anotada con el nombre del frame actual
        logging.info(f"Guardando imagen {self.frame_count:04d}")
        frame_filename = os.path.join(self.config.OUTPUT_FOLDER, f"frame_{self.frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)

        # Obtener fecha y hora actuales
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H:%M:%S")

        # Agregar resultados al buffer
        self.csv_buffer.append(
            [self.frame_count, current_date, current_time, detected_objects, confidences, coordinates, areas]
        )

        # Escribir en el CSV si se alcanza el tamaño del buffer
        if len(self.csv_buffer) >= self.config.CSV_BUFFER_SIZE:
            self.write_csv_buffer()

    def write_csv_buffer(self):
        try:
            with open(self.csv_filepath, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(self.csv_buffer)
            logging.info(f"Escrito {len(self.csv_buffer)} entradas al CSV.")
            self.csv_buffer = []
        except Exception as e:
            logging.error(f"Error al escribir en el CSV: {e}")


class DetectionApp:
    """Clase principal para la aplicación de detección de objetos"""

    def __init__(self):
        self.config = Config()
        self.frame_queue = Queue()
        self.motion_event = threading.Event()
        self.stop_event = threading.Event()
        self.detector_thread = None
        self.frame_capture_thread = None

        # Crear carpeta de salida si no existe
        if not os.path.exists(self.config.OUTPUT_FOLDER):
            os.makedirs(self.config.OUTPUT_FOLDER)
            logging.info(f"Carpeta de salida creada: {self.config.OUTPUT_FOLDER}")

        # Inicializar el archivo CSV
        self.csv_filepath = os.path.join(self.config.OUTPUT_FOLDER, self.config.CSV_FILENAME)
        self.init_csv()

    def init_csv(self):
        # Escribir la cabecera del CSV si el archivo no existe
        if not os.path.isfile(self.csv_filepath):
            try:
                with open(self.csv_filepath, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            "Frame",
                            "Fecha",
                            "Hora",
                            "Objetos Detectados",
                            "Confidencias",
                            "Coordenadas",
                            "Tamaño del área",
                        ]
                    )
                logging.info(f"Archivo CSV inicializado: {self.csv_filepath}")
            except Exception as e:
                logging.error(f"Error al inicializar el CSV: {e}")

    def start(self):
        # Iniciar hilo de captura de frames
        self.frame_capture_thread = FrameCapture(self.config, self.frame_queue, self.motion_event, self.stop_event)
        self.frame_capture_thread.start()
        logging.info("FrameCaptureThread iniciado.")

        # Iniciar hilo de detección
        self.detector_thread = DetectorThread(self.config, self.frame_queue, self.motion_event, self.stop_event)
        self.detector_thread.start()
        logging.info("DetectorThread iniciado.")

    def stop(self):
        logging.info("Deteniendo la aplicación...")
        self.stop_event.set()

        # Detener el hilo de captura de frames
        if self.frame_capture_thread.is_alive():
            self.frame_capture_thread.stop()
            self.frame_capture_thread.join()
            logging.info("FrameCaptureThread detenido.")

        # Detener el hilo de detección
        if self.detector_thread.is_alive():
            self.detector_thread.join()
            logging.info("DetectorThread detenido.")

        logging.info("Aplicación detenida correctamente.")

    def signal_handler(self, sig, frame):
        self.stop()

    def run(self):
        # Manejar señal para Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        logging.info("Iniciando la aplicación de detección.")
        self.start()

        # Esperar hasta que se indique detenerse
        self.stop_event.wait()
        self.stop()


if __name__ == "__main__":
    app = DetectionApp()
    app.run()
