import csv
import functools
import logging
import os
import signal
import threading
from contextlib import contextmanager
from datetime import datetime
from queue import Empty, Queue
from typing import List, Optional, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# Configuración del registro (logging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    # Para agregar un archivo de log, descomenta las líneas siguientes:
    # handlers=[
    #     logging.StreamHandler(),
    #     logging.FileHandler("application.log")
    # ]
)


def handle_exceptions(func):
    """Decorador para manejar y registrar excepciones en métodos."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error en {func.__name__}: {e}", exc_info=True)
            raise

    return wrapper


@contextmanager
def managed_camera(config: "Config"):
    """Context manager para manejar la cámara."""
    picam2 = Picamera2()
    try:
        camera_config = picam2.create_preview_configuration(main={"size": config.CAMERA_RESOLUTION})
        picam2.configure(camera_config)
        picam2.start()
        logging.info("Cámara iniciada.")
        yield picam2
    finally:
        picam2.stop()
        logging.info("Cámara detenida y liberada.")


class Config:
    ENABLE_VISUALIZATION: bool = False      # Controla si se muestra la ventana de video (True/False)
    DETECTION_LIST: List[str] = ["person"]  # Lista de objetos a detectar
    CONFIDENCE_THRESHOLD: float = 0.15       # Confianza mínima para detectar objetos
    OUTPUT_FOLDER: str = "output"           # Carpeta de salida para imágenes y CSV
    CSV_FILENAME: str = "detections.csv"    # Nombre del archivo CSV para almacenar resultados
    MIN_MOTION_AREA: int = 5000             # Área mínima de movimiento para considerar que hay cambio
    RESIZE_WIDTH: int = 640                 # Ancho al que se redimensionará la imagen para detección de movimiento
    CAMERA_RESOLUTION: Tuple[int, int] = (640, 360) # Resolución de la cámara (ancho, alto)
    MODEL_NAME: str = "yolov8x.pt"          # Modelo YOLOv8 a utilizar
    CSV_BUFFER_SIZE: int = 2                # Número de entradas antes de escribir en el CSV
    MOTION_DETECTION_COOLDOWN: int = 5      # Segundos de espera después de detectar movimiento
    QUEUE_MAXSIZE: int = 5                  # Tamaño máximo de la cola de frames
    BACKGROUND_SUBTRACTOR: dict = {         # Configuración para createBackgroundSubtractorKNN
        "history": 500,
        "dist2Threshold": 400.0,
        "detectShadows": False,
    }


class ObjectDetector:
    """Clase para manejar la detección de objetos con YOLOv8"""

    def __init__(self, config: Config):
        self.config = config
        self.model = self.load_model()
        self.classes_ids: Optional[List[int]] = self.get_classes_ids()

    @handle_exceptions
    def load_model(self) -> YOLO:
        """Carga el modelo YOLO especificado."""
        model = YOLO(self.config.MODEL_NAME)
        logging.info(f"Modelo YOLO '{self.config.MODEL_NAME}' cargado y listo para usar.")
        return model

    def get_classes_ids(self) -> Optional[List[int]]:
        """Obtiene los IDs de las clases a detectar."""
        if not self.config.DETECTION_LIST:
            return None
        classes_ids = []
        for object_name in self.config.DETECTION_LIST:
            try:
                class_id = list(self.model.names.values()).index(object_name)
                classes_ids.append(class_id)
            except ValueError:
                logging.warning(f"Objeto '{object_name}' no encontrado en las clases del modelo.")
        return classes_ids

    @handle_exceptions
    def detect_objects(self, frame: np.ndarray) -> List:
        """Realiza la detección de objetos en el frame proporcionado."""
        results = self.model(frame, classes=self.classes_ids, conf=self.config.CONFIDENCE_THRESHOLD)
        return results

    def filter_detections(self, results: List) -> Tuple[List[str], List[float], List[List[int]], List[Tuple[int, int]]]:
        """Filtra las detecciones según la confianza y extrae información relevante."""
        detected_objects: List[str] = []
        confidences: List[float] = []
        coordinates: List[List[int]] = []
        areas: List[Tuple[int, int]] = []

        # Verificar si hay detecciones
        if results and results[0].boxes is not None and len(results[0].boxes.data) > 0:
            for result in results[0].boxes.data:
                object_class = int(result[5])
                confidence = float(result[4])
                object_name = self.model.names.get(object_class, "Unknown")

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

    def __init__(self, config: Config, frame_queue: Queue, motion_event: threading.Event, stop_event: threading.Event):
        super().__init__()
        self.config = config
        self.frame_queue = frame_queue
        self.motion_event = motion_event
        self.stop_event = stop_event

    @handle_exceptions
    def run(self):
        """Método principal que ejecuta el hilo de captura de frames."""
        last_motion_time: Optional[datetime] = None
        with managed_camera(self.config) as picam2:
            backSub = cv2.createBackgroundSubtractorKNN(
                history=self.config.BACKGROUND_SUBTRACTOR["history"],
                dist2Threshold=self.config.BACKGROUND_SUBTRACTOR["dist2Threshold"],
                detectShadows=self.config.BACKGROUND_SUBTRACTOR["detectShadows"],
            )
            logging.info("Sustractor de fondo KNN inicializado.")

            while not self.stop_event.is_set():
                # Capturar un frame de la cámara
                frame: np.ndarray = picam2.capture_array()

                # Aplicar el sustractor de fondo para detectar movimiento
                fg_mask: np.ndarray = backSub.apply(frame)

                # Operaciones morfológicas para reducir el ruido
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

                # Encontrar contornos
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                motion_detected: bool = False
                for contour in contours:
                    if cv2.contourArea(contour) < self.config.MIN_MOTION_AREA:
                        continue
                    motion_detected = True
                    break

                current_time: datetime = datetime.now()

                if motion_detected:
                    if (
                        last_motion_time is None
                        or (current_time - last_motion_time).total_seconds() > self.config.MOTION_DETECTION_COOLDOWN
                    ):
                        logging.info("Movimiento detectado. Encolando frame para detección.")
                        try:
                            self.frame_queue.put(frame, timeout=1)
                            last_motion_time = current_time
                            self.motion_event.set()
                        except Exception as e:
                            logging.error(f"Error al encolar el frame: {e}")
                else:
                    # Si ha pasado el periodo de enfriamiento, resetear el evento
                    if (
                        last_motion_time
                        and (current_time - last_motion_time).total_seconds() > self.config.MOTION_DETECTION_COOLDOWN
                    ):
                        self.motion_event.clear()
                        last_motion_time = None


class DetectorThread(threading.Thread):
    """Hilo para la detección de objetos"""

    def __init__(self, config: Config, frame_queue: Queue, motion_event: threading.Event, stop_event: threading.Event):
        super().__init__()  # No queremos daemon threads
        self.config = config
        self.frame_queue = frame_queue
        self.motion_event = motion_event
        self.stop_event = stop_event
        self.frame_count: int = 0
        self.csv_buffer: List[List] = []
        self.csv_filepath: str = os.path.join(self.config.OUTPUT_FOLDER, self.config.CSV_FILENAME)
        self.detector = ObjectDetector(self.config)

    @handle_exceptions
    def run(self) -> None:
        """Método principal que ejecuta el hilo de detección."""
        try:
            while not self.stop_event.is_set():
                frame = self.get_frame()
                if frame is not None:
                    self.process_frame(frame)
        except Exception as e:
            logging.error(f"Error en DetectorThread: {e}")
        finally:
            self.cleanup()

    def get_frame(self) -> Optional[np.ndarray]:
        """Obtiene un frame de la cola, manejando excepciones."""
        try:
            frame = self.frame_queue.get(timeout=1)
            return frame
        except Empty:
            return None

    def process_frame(self, frame: np.ndarray) -> None:
        """Procesa el frame para detectar objetos y guarda los resultados."""
        try:
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
            self.display_results(annotated_frame)

            self.frame_count += 1

        except Exception as e:
            logging.error(f"Error durante el procesamiento del frame: {e}")

    def display_results(self, annotated_frame: np.ndarray) -> None:
        """Muestra los resultados en pantalla si la visualización está habilitada."""
        if self.config.ENABLE_VISUALIZATION:
            cv2.imshow("Detección en Tiempo Real", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("Tecla 'q' presionada. Deteniendo la aplicación.")
                self.stop_event.set()

    def cleanup(self) -> None:
        """Limpia los recursos al finalizar el hilo."""
        if self.config.ENABLE_VISUALIZATION:
            cv2.destroyAllWindows()
        if self.csv_buffer:
            self.write_csv_buffer()
        logging.info("DetectorThread finalizado.")

    @handle_exceptions
    def save_results(
        self,
        annotated_frame: np.ndarray,
        detected_objects: List[str],
        confidences: List[float],
        coordinates: List[List[int]],
        areas: List[Tuple[int, int]],
    ) -> None:
        """Guarda los resultados de la detección en imágenes y CSV."""
        # Guardar la imagen anotada con el nombre del frame actual
        logging.info(f"Guardando imagen {self.frame_count:04d}")
        frame_filename: str = os.path.join(self.config.OUTPUT_FOLDER, f"frame_{self.frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, annotated_frame)

        # Obtener fecha y hora actuales
        now: datetime = datetime.now()
        current_date: str = now.strftime("%Y-%m-%d")
        current_time: str = now.strftime("%H:%M:%S")

        # Agregar resultados al buffer
        self.csv_buffer.append(
            [
                self.frame_count,
                current_date,
                current_time,
                detected_objects,
                confidences,
                coordinates,
                areas,
            ]
        )

        # Escribir en el CSV si se alcanza el tamaño del buffer
        if len(self.csv_buffer) >= self.config.CSV_BUFFER_SIZE:
            self.write_csv_buffer()

    def write_csv_buffer(self) -> None:
        """Escribe el buffer de resultados en el archivo CSV."""
        try:
            with open(self.csv_filepath, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(self.csv_buffer)
            logging.info(f"Escrito {len(self.csv_buffer)} entradas al CSV.")
            self.csv_buffer.clear()
        except Exception as e:
            logging.error(f"Error al escribir en el CSV: {e}")


class DetectionApp:
    """Clase principal para la aplicación de detección de objetos"""

    def __init__(self):
        self.config = Config()
        self.frame_queue: Queue = Queue(maxsize=self.config.QUEUE_MAXSIZE)
        self.motion_event: threading.Event = threading.Event()
        self.stop_event: threading.Event = threading.Event()
        self.detector_thread: Optional[DetectorThread] = None
        self.frame_capture_thread: Optional[FrameCapture] = None

        # Crear carpeta de salida si no existe
        try:
            os.makedirs(self.config.OUTPUT_FOLDER, exist_ok=True)
            logging.info(f"Carpeta de salida asegurada: {self.config.OUTPUT_FOLDER}")
        except Exception as e:
            logging.error(f"Error al crear la carpeta de salida '{self.config.OUTPUT_FOLDER}': {e}")
            raise

        # Inicializar el archivo CSV
        self.csv_filepath: str = os.path.join(self.config.OUTPUT_FOLDER, self.config.CSV_FILENAME)
        self.init_csv()

    @handle_exceptions
    def init_csv(self) -> None:
        """Inicializa el archivo CSV con la cabecera si no existe."""
        if not os.path.isfile(self.csv_filepath):
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

    @handle_exceptions
    def start(self) -> None:
        """Inicia los hilos de captura de frames y detección de objetos."""
        # Iniciar hilo de captura de frames
        self.frame_capture_thread = FrameCapture(
            config=self.config, frame_queue=self.frame_queue, motion_event=self.motion_event, stop_event=self.stop_event
        )
        self.frame_capture_thread.start()
        logging.info("FrameCaptureThread iniciado.")

        # Iniciar hilo de detección
        self.detector_thread = DetectorThread(
            config=self.config, frame_queue=self.frame_queue, motion_event=self.motion_event, stop_event=self.stop_event
        )
        self.detector_thread.start()
        logging.info("DetectorThread iniciado.")

    def stop(self) -> None:
        """Detiene la aplicación y los hilos en ejecución."""
        logging.info("Deteniendo la aplicación...")
        self.stop_event.set()

        # Detener el hilo de captura de frames
        if self.frame_capture_thread and self.frame_capture_thread.is_alive():
            try:
                self.frame_capture_thread.stop()
                self.frame_capture_thread.join(timeout=5)
                logging.info("FrameCaptureThread detenido.")
            except Exception as e:
                logging.error(f"Error al detener FrameCaptureThread: {e}")

        # Detener el hilo de detección
        if self.detector_thread and self.detector_thread.is_alive():
            try:
                self.detector_thread.join(timeout=5)
                logging.info("DetectorThread detenido.")
            except Exception as e:
                logging.error(f"Error al detener DetectorThread: {e}")

        logging.info("Aplicación detenida correctamente.")

    def signal_handler(self, sig: int, frame: Optional[object]) -> None:
        """Maneja las señales recibidas, como Ctrl+C."""
        logging.info("Señal de terminación recibida. Iniciando la detención de la aplicación.")
        self.stop()

    def run(self) -> None:
        """Ejecuta la aplicación de detección de objetos."""
        # Manejar señal para Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        logging.info("Iniciando la aplicación de detección.")
        self.start()

        # Esperar hasta que se indique detenerse
        try:
            while not self.stop_event.is_set():
                self.stop_event.wait(timeout=1)
        except Exception as e:
            logging.error(f"Error en el bucle principal: {e}")
            self.stop()


if __name__ == "__main__":
    try:
        app = DetectionApp()
        app.run()
    except Exception as e:
        logging.critical(f"Error crítico en la aplicación: {e}", exc_info=True)
