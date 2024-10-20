"""
detection.py

Este módulo contiene las clases relacionadas con la detección de objetos utilizando YOLOv8, así como los hilos para la captura de frames y el procesamiento de detección.
"""

import csv
import logging
import os
import tempfile
import threading
from datetime import datetime
from queue import Empty, Queue
from typing import List, Optional, Tuple

import cv2
import numpy as np
from config import Config
from telegram import TelegramNotifier
from ultralytics import YOLO
from utils import handle_exceptions, managed_camera


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
        logging.debug(f"Modelo YOLO '{self.config.MODEL_NAME}' cargado y listo para usar.")
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
            logging.debug("No se detectaron objetos en este frame.")

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
        self.telegram_notifier = TelegramNotifier(self.config)  # Inicializar dentro del hilo
        try:
            with managed_camera(self.config) as camera:
                backSub = cv2.createBackgroundSubtractorKNN(
                    history=self.config.BACKGROUND_SUBTRACTOR["history"],
                    dist2Threshold=self.config.BACKGROUND_SUBTRACTOR["dist2Threshold"],
                    detectShadows=self.config.BACKGROUND_SUBTRACTOR["detectShadows"],
                )
                logging.debug("Sustractor de fondo KNN inicializado.")

                while not self.stop_event.is_set():
                    # Capturar un frame de la cámara
                    if self.config.MODE == "prod":
                        frame: np.ndarray = camera.capture_array()
                    else:
                        ret, frame = camera.read()
                        if not ret:
                            logging.error("No se pudo leer el frame de la webcam.")
                            continue

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
                            logging.warning("Movimiento detectado. Encolando frame para detección.")
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
                            and (current_time - last_motion_time).total_seconds()
                            > self.config.MOTION_DETECTION_COOLDOWN
                        ):
                            self.motion_event.clear()
                            last_motion_time = None
        except Exception as e:
            logging.error(f"Error en FrameCapture: {e}")
            self.telegram_notifier.send_message(f"Error en FrameCapture: {e}")
            self.stop_event.set()


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
        self.telegram_notifier = TelegramNotifier(self.config)

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
            self.telegram_notifier.send_message(f"Error en DetectorThread: {e}")
            self.stop_event.set()
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
            self.telegram_notifier.send_message(f"Error durante el procesamiento del frame: {e}")

    def display_results(self, annotated_frame: np.ndarray) -> None:
        """Muestra los resultados en pantalla si la visualización está habilitada."""
        if self.config.ENABLE_VISUALIZATION:
            cv2.imshow("Detección en Tiempo Real", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.warning("Tecla 'q' presionada. Deteniendo la aplicación.")
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
        try:
            if self.config.DELETE_SENT_IMAGES:
                with tempfile.NamedTemporaryFile(
                    suffix=".jpg", delete=False, dir=self.config.OUTPUT_FOLDER
                ) as tmp_file:
                    frame_filename = tmp_file.name
                    cv2.imwrite(frame_filename, annotated_frame)
            else:
                # Guardar la imagen en la carpeta de salida
                frame_filename = os.path.join(self.config.OUTPUT_FOLDER, f"frame_{self.frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, annotated_frame)

            logging.debug(f"Imagen guardada: {frame_filename}")

            # Formatear las coordenadas para el mensaje
            coordenadas_formateadas = "; ".join(
                [f"({x_min}, {y_min}, {x_max}, {y_max})" for x_min, y_min, x_max, y_max in coordinates]
            )

            # Obtener la hora específica
            now: datetime = datetime.now()
            hora_especifica: str = now.strftime("%Y-%m-%d %H:%M:%S")

            # Construir el caption con información adicional
            caption = (
                f"Frame {self.frame_count:04d}\n"
                f"Hora: {hora_especifica}\n"
                f"Objetos Detectados: {', '.join(detected_objects)}\n"
                f"Coordenadas: {coordenadas_formateadas}"
            )
            # Enviar la imagen al canal de imágenes en Telegram como documento
            self.telegram_notifier.send_document(frame_filename, caption=caption)

            # Eliminar la imagen si está configurado
            if self.config.DELETE_SENT_IMAGES:
                try:
                    os.remove(frame_filename)
                    logging.debug(f"Imagen eliminada: {frame_filename}")
                except Exception as e:
                    logging.error(f"Error al eliminar la imagen {frame_filename}: {e}")
                    self.telegram_notifier.send_message(f"Error al eliminar la imagen {frame_filename}: {e}")

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

        except Exception as e:
            logging.error(f"Error al guardar resultados: {e}")
            self.telegram_notifier.send_message(f"Error al guardar resultados: {e}")

    def write_csv_buffer(self) -> None:
        """Escribe el buffer de resultados en el archivo CSV."""
        try:
            with open(self.csv_filepath, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(self.csv_buffer)
            logging.debug(f"Escrito {len(self.csv_buffer)} entradas al CSV.")
            self.csv_buffer.clear()
        except Exception as e:
            logging.error(f"Error al escribir en el CSV: {e}")
            self.telegram_notifier.send_message(f"Error al escribir en el CSV: {e}")
