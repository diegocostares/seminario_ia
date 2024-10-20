"""
main.py

Este es el punto de entrada de la aplicación de detección de objetos. Inicializa la configuración, configura el logging, y orquesta la ejecución de los hilos para la captura de frames y la detección de objetos.
"""

import csv
import logging
import os
import signal
import sys
import threading
from queue import Queue
from typing import Optional

from config import Config
from detection import DetectorThread, FrameCapture, handle_exceptions
from telegram import TelegramNotifier
from utils import setup_logging


class DetectionApp:
    """Clase principal para la aplicación de detección de objetos"""

    def __init__(self):
        self.config = Config()

        # Configurar el logging incluyendo el handler de consola
        setup_logging(self.config)

        # Inicializar el notificador de Telegram
        self.telegram_notifier = TelegramNotifier(self.config)

        self.frame_queue = Queue(maxsize=self.config.QUEUE_MAXSIZE)
        self.motion_event = threading.Event()
        self.stop_event = threading.Event()
        self.detector_thread: Optional[DetectorThread] = None
        self.frame_capture_thread: Optional[FrameCapture] = None

        # Verificar que las variables de entorno de Telegram estén configuradas
        if not all(
            [
                self.config.TELEGRAM_BOT_TOKEN,
                self.config.TELEGRAM_IMAGE_CHANNEL_ID,
                self.config.TELEGRAM_LOG_CHANNEL_ID,
            ]
        ):
            logging.error("Las variables de entorno de Telegram no están configuradas correctamente.")
            self.telegram_notifier.send_message("Error: Variables de entorno de Telegram faltantes.")
            raise EnvironmentError("Variables de entorno de Telegram faltantes.")

        # Crear carpeta de salida si no existe
        try:
            os.makedirs(self.config.OUTPUT_FOLDER, exist_ok=True)
            logging.debug(f"Carpeta de salida asegurada: {self.config.OUTPUT_FOLDER}")
        except Exception as e:
            logging.error(f"Error al crear la carpeta de salida '{self.config.OUTPUT_FOLDER}': {e}")
            self.telegram_notifier.send_message(
                f"Error al crear la carpeta de salida '{self.config.OUTPUT_FOLDER}': {e}"
            )
            raise

        # Inicializar el archivo CSV
        self.csv_filepath = os.path.join(self.config.OUTPUT_FOLDER, self.config.CSV_FILENAME)
        self.init_csv()

    @handle_exceptions
    def init_csv(self) -> None:
        """Inicializa el archivo CSV con la cabecera si no existe."""
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
                logging.debug(f"Archivo CSV inicializado: {self.csv_filepath}")
            except Exception as e:
                logging.error(f"Error al inicializar el archivo CSV: {e}")
                self.telegram_notifier.send_message(f"Error al inicializar el archivo CSV: {e}")
                raise

    @handle_exceptions
    def start(self) -> None:
        """Inicia los hilos de captura de frames y detección de objetos."""
        # Iniciar hilo de captura de frames
        self.frame_capture_thread = FrameCapture(
            config=self.config, frame_queue=self.frame_queue, motion_event=self.motion_event, stop_event=self.stop_event
        )
        self.frame_capture_thread.start()
        logging.debug("FrameCaptureThread iniciado.")

        # Iniciar hilo de detección
        self.detector_thread = DetectorThread(
            config=self.config, frame_queue=self.frame_queue, motion_event=self.motion_event, stop_event=self.stop_event
        )
        self.detector_thread.start()
        logging.debug("DetectorThread iniciado.")

    @handle_exceptions
    def stop(self) -> None:
        """Detiene la aplicación y los hilos en ejecución."""
        logging.critical("Deteniendo la aplicación...")
        self.stop_event.set()

        # Detener el hilo de captura de frames
        if self.frame_capture_thread and self.frame_capture_thread.is_alive():
            try:
                self.frame_capture_thread.join(timeout=5)
                logging.debug("FrameCaptureThread detenido.")
            except Exception as e:
                logging.error(f"Error al detener FrameCaptureThread: {e}")
                self.telegram_notifier.send_message(f"Error al detener FrameCaptureThread: {e}")

        # Detener el hilo de detección
        if self.detector_thread and self.detector_thread.is_alive():
            try:
                self.detector_thread.join(timeout=5)
                logging.debug("DetectorThread detenido.")
            except Exception as e:
                logging.error(f"Error al detener DetectorThread: {e}")
                self.telegram_notifier.send_message(f"Error al detener DetectorThread: {e}")

        # Enviar mensaje de finalización a Telegram
        try:
            self.telegram_notifier.send_message("La aplicación de detección de objetos ha finalizado correctamente.")
        except Exception as e:
            logging.error(f"Error al enviar mensaje de finalización a Telegram: {e}")

        logging.debug("Aplicación detenida correctamente.")

    def signal_handler(self, sig: int, frame: Optional[object]) -> None:
        """Maneja las señales recibidas, como Ctrl+C."""
        logging.warning("Señal de terminación recibida. Iniciando la detención de la aplicación.")
        self.stop()

    def run(self) -> None:
        """Ejecuta la aplicación de detección de objetos."""
        # Manejar señal para Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        logging.debug("Iniciando la aplicación de detección.")
        self.start()
        self.telegram_notifier.send_message("La aplicación ha iniciado correctamente.")

        # Esperar hasta que se indique detenerse
        try:
            while not self.stop_event.is_set():
                self.stop_event.wait(timeout=1)
        except Exception as e:
            logging.error(f"Error en el bucle principal: {e}")
            self.telegram_notifier.send_message(f"Error en el bucle principal: {e}")
            self.stop()


if __name__ == "__main__":
    try:
        app = DetectionApp()
        app.run()
    except Exception as e:
        logging.critical(f"Error crítico en la aplicación: {e}", exc_info=True)
        # Enviar el log de error al canal de logs
        try:
            if "app" in locals() and app.telegram_notifier:
                app.telegram_notifier.send_message(f"Error crítico en la aplicación: {e}")
        except Exception as notifier_error:
            logging.error(f"Error al enviar mensaje crítico a Telegram: {notifier_error}")
        sys.exit(1)
