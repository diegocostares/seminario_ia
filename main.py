# main.py

import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from queue import Empty, Full, Queue
from typing import Optional

import cv2
import numpy as np
from config import Config
from telegram import TelegramNotifier
from utils import handle_exceptions, managed_camera, setup_logging


class MotionDetectorApp:
    """Clase principal para la aplicación de detección de movimiento"""

    def __init__(self):
        self.config = Config()

        # Configurar el logging
        setup_logging(self.config)

        # Inicializar el notificador de Telegram
        self.telegram_notifier = TelegramNotifier(self.config)

        self.frame_queue = Queue(maxsize=self.config.QUEUE_MAXSIZE)
        self.motion_event = threading.Event()
        self.stop_event = threading.Event()
        self.frame_capture_thread: Optional[FrameCaptureThread] = None
        self.motion_handler_thread: Optional[MotionHandlerThread] = None

        # Verificar configuración de Telegram
        if not all(
            [self.config.TELEGRAM_BOT_TOKEN, self.config.TELEGRAM_IMAGE_CHANNEL_ID, self.config.TELEGRAM_LOG_CHANNEL_ID]
        ):
            logging.error("Variables de entorno de Telegram faltantes.")
            self.telegram_notifier.send_message("Error: Variables de entorno de Telegram faltantes.")
            raise EnvironmentError("Variables de entorno de Telegram faltantes.")

        # Crear carpeta de salida si no existe
        os.makedirs(self.config.OUTPUT_FOLDER, exist_ok=True)

    @handle_exceptions
    def start(self) -> None:
        """Inicia los hilos de captura de frames y manejo de movimiento."""
        self.frame_capture_thread = FrameCaptureThread(
            config=self.config, frame_queue=self.frame_queue, motion_event=self.motion_event, stop_event=self.stop_event
        )
        self.frame_capture_thread.start()
        logging.debug("FrameCaptureThread iniciado.")

        self.motion_handler_thread = MotionHandlerThread(
            config=self.config,
            frame_queue=self.frame_queue,
            motion_event=self.motion_event,
            stop_event=self.stop_event,
            telegram_notifier=self.telegram_notifier,
        )
        self.motion_handler_thread.start()
        logging.debug("MotionHandlerThread iniciado.")

    @handle_exceptions
    def stop(self) -> None:
        """Detiene la aplicación y los hilos en ejecución."""
        logging.critical("Deteniendo la aplicación...")
        self.stop_event.set()

        # Detener el hilo de captura de frames
        if self.frame_capture_thread and self.frame_capture_thread.is_alive():
            self.frame_capture_thread.join(timeout=5)
            logging.debug("FrameCaptureThread detenido.")

        # Detener el hilo de manejo de movimiento
        if self.motion_handler_thread and self.motion_handler_thread.is_alive():
            self.motion_handler_thread.join(timeout=5)
            logging.debug("MotionHandlerThread detenido.")

        # Enviar mensaje de finalización a Telegram
        self.telegram_notifier.send_message("La aplicación de detección de movimiento ha finalizado correctamente.")
        logging.debug("Aplicación detenida correctamente.")

    def signal_handler(self, sig: int, frame: Optional[object]) -> None:
        """Maneja las señales recibidas, como Ctrl+C."""
        logging.warning("Señal de terminación recibida. Iniciando la detención de la aplicación.")
        self.stop()

    def run(self) -> None:
        """Ejecuta la aplicación de detección de movimiento."""
        signal.signal(signal.SIGINT, self.signal_handler)
        logging.debug("Iniciando la aplicación de detección de movimiento.")
        self.start()
        self.telegram_notifier.send_message("La aplicación ha iniciado correctamente.")

        try:
            while not self.stop_event.is_set():
                self.stop_event.wait(timeout=1)
        except Exception as e:
            logging.error(f"Error en el bucle principal: {e}")
            self.telegram_notifier.send_message(f"Error en el bucle principal: {e}")
            self.stop()


class FrameCaptureThread(threading.Thread):
    """Hilo para capturar frames de la cámara y detectar movimiento"""

    def __init__(self, config: Config, frame_queue: Queue, motion_event: threading.Event, stop_event: threading.Event):
        super().__init__()
        self.config = config
        self.frame_queue = frame_queue
        self.motion_event = motion_event
        self.stop_event = stop_event
        self.last_capture_time = time.time()

    @handle_exceptions
    def run(self):
        try:
            with managed_camera(self.config) as camera:
                backSub = cv2.createBackgroundSubtractorKNN(
                    history=self.config.BACKGROUND_SUBTRACTOR["history"],
                    dist2Threshold=self.config.BACKGROUND_SUBTRACTOR["dist2Threshold"],
                    detectShadows=self.config.BACKGROUND_SUBTRACTOR["detectShadows"],
                )

                polygon_points = np.array(self.config.POLYGON_POINTS, np.int32).reshape((-1, 1, 2))

                while not self.stop_event.is_set():
                    current_time = time.time()

                    if current_time - self.last_capture_time < self.config.DETECTION_INTERVAL:
                        time.sleep(0.1)
                        continue

                    if self.config.MODE == "prod":
                        frame: np.ndarray = camera.capture_array()
                    else:
                        ret, frame = camera.read()
                        if not ret:
                            logging.error("No se pudo leer el frame de la cámara.")
                            continue

                    fg_mask: np.ndarray = backSub.apply(frame)
                    mask_polygon = np.zeros_like(fg_mask)
                    cv2.fillPoly(mask_polygon, [polygon_points], 255)
                    fg_mask = cv2.bitwise_and(fg_mask, mask_polygon)

                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    motion_detected = any(
                        cv2.contourArea(contour) > self.config.MIN_MOTION_AREA for contour in contours
                    )

                    if motion_detected:
                        try:
                            self.frame_queue.put_nowait(frame)
                            self.motion_event.set()
                            self.last_capture_time = current_time
                            logging.debug("Frame encolado exitosamente.")
                        except Full:
                            logging.warning("La cola de frames está llena. Descartando frame actual.")
                            time.sleep(0.2)

        except Exception as e:
            logging.error(f"Error en FrameCaptureThread: {e}")
            self.stop_event.set()


class MotionHandlerThread(threading.Thread):
    """Hilo para manejar la detección de movimiento y enviar imágenes a Telegram"""

    def __init__(
        self,
        config: Config,
        frame_queue: Queue,
        motion_event: threading.Event,
        stop_event: threading.Event,
        telegram_notifier: TelegramNotifier,
    ):
        super().__init__()
        self.config = config
        self.frame_queue = frame_queue
        self.motion_event = motion_event
        self.stop_event = stop_event
        self.telegram_notifier = telegram_notifier
        self.frame_count = 0
        self.last_sent_time = datetime.min

    @handle_exceptions
    def run(self):
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=1)
                    self.handle_motion(frame)
                except Empty:
                    continue
        except Exception as e:
            logging.error(f"Error en MotionHandlerThread: {e}")
            self.stop_event.set()

    def handle_motion(self, frame: np.ndarray) -> None:
        """Procesa el frame para capturar y enviar la imagen al detectar movimiento."""
        try:
            timestamp = datetime.now()
            if (timestamp - self.last_sent_time).total_seconds() < self.config.DETECTION_INTERVAL:
                logging.info("Intervalo de envío no cumplido. Ignorando frame.")
                return

            filename = timestamp.strftime("%d-%m") + f"-frame{self.frame_count:04d}.jpg"
            file_path = os.path.join(self.config.OUTPUT_FOLDER, filename)

            if self.config.SAVE_IMAGES:
                cv2.imwrite(file_path, frame)
                logging.info(f"Imagen de movimiento guardada: {file_path}")

                if os.path.exists(file_path):
                    caption = f"Movimiento detectado a las {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                    self.telegram_notifier.send_document(file_path, caption=caption)
                    logging.info(f"Imagen de movimiento enviada a Telegram: {file_path}")

                    if self.config.DELETE_SENT_IMAGES:
                        os.remove(file_path)
                        logging.debug(f"Imagen eliminada después de envío: {file_path}")
            else:
                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    logging.error("Error al codificar el frame a JPEG.")
                    self.telegram_notifier.send_message("Error al codificar el frame a JPEG.")
                    return

                image_bytes = buffer.tobytes()
                caption = f"Movimiento detectado a las {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                self.telegram_notifier.send_image_as_document(image_bytes, caption=caption, filename=filename)
                logging.info("Imagen de movimiento enviada a Telegram sin guardar localmente.")

            self.last_sent_time = timestamp
            self.frame_count += 1

        except Exception as e:
            logging.error(f"Error al manejar el movimiento: {e}")
            self.telegram_notifier.send_message(f"Error al manejar el movimiento: {e}")


if __name__ == "__main__":
    try:
        app = MotionDetectorApp()
        app.run()
    except Exception as e:
        logging.critical(f"Error crítico en la aplicación: {e}", exc_info=True)
        if "app" in locals() and app.telegram_notifier:
            try:
                app.telegram_notifier.send_message(f"Error crítico en la aplicación: {e}")
            except Exception as notifier_error:
                logging.error(f"Error al enviar mensaje crítico a Telegram: {notifier_error}")
        sys.exit(1)
