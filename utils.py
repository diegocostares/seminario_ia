"""
utils.py

Este módulo proporciona funciones utilitarias, incluyendo un context manager para manejar la cámara de manera segura y un decorador para manejar y registrar excepciones en métodos.
También incluye la función `setup_logging` para configurar el sistema de logging.
"""

import functools
import logging
from contextlib import contextmanager

import cv2
from config import Config


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
def managed_camera(config: Config):
    """Context manager para manejar la cámara."""
    if config.MODE == "prod":
        # Usar la cámara de Raspberry Pi
        from picamera2 import Picamera2

        picam2 = Picamera2()
        try:
            camera_config = picam2.create_preview_configuration(main={"size": config.CAMERA_RESOLUTION})
            picam2.configure(camera_config)
            picam2.start()
            logging.critical("Cámara de Raspberry Pi iniciada.")
            yield picam2
        finally:
            picam2.stop()
            logging.critical("Cámara de Raspberry Pi detenida y liberada.")
    else:
        # Usar la webcam del PC
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
        if not cap.isOpened():
            logging.error("No se pudo abrir la cámara web.")
            raise Exception("No se pudo abrir la cámara web.")
        logging.info("Webcam del PC iniciada.")
        try:
            yield cap
        finally:
            cap.release()
            logging.info("Webcam del PC detenida y liberada.")


def setup_logging(config: Config):
    """
    Configura el sistema de logging, incluyendo el handler para la consola.
    """
    # Obtener el logger raíz
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Establecer nivel más bajo para permitir que los handlers filtren

    # Limpiar cualquier handler existente
    if logger.hasHandlers():
        logger.handlers.clear()

    # Configurar el handler de consola
    console_handler = logging.StreamHandler()
    console_level = getattr(logging, config.LOG_CONSOLE_LEVEL.upper(), logging.WARNING)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
