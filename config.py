"""
config.py

Este módulo define la clase Config que almacena todas las configuraciones necesarias para la aplicación de detección de objetos. Incluye configuraciones de cámara, detección, Telegram y opciones de visualización.
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()


class Config:
    ENABLE_VISUALIZATION: bool = False  # Controla si se muestra la ventana de video (True/False)
    DETECTION_LIST: List[str] = ["person"]  # Lista de objetos a detectar
    CONFIDENCE_THRESHOLD: float = 0.2  # Confianza mínima para detectar objetos
    OUTPUT_FOLDER: str = "output"  # Carpeta de salida para imágenes y CSV
    CSV_FILENAME: str = "detections.csv"  # Nombre del archivo CSV para almacenar resultados
    MIN_MOTION_AREA: int = 5000  # Área mínima de movimiento para considerar que hay cambio
    CAMERA_RESOLUTION: Tuple[int, int] = (640, 360)  # Resolución de la cámara (ancho, alto)
    MODEL_NAME: str = "yolov8x.pt"  # Modelo YOLOv8 a utilizar
    CSV_BUFFER_SIZE: int = 2  # Número de entradas antes de escribir en el CSV
    MOTION_DETECTION_COOLDOWN: int = 5  # Segundos de espera después de detectar movimiento
    QUEUE_MAXSIZE: int = 5  # Tamaño máximo de la cola de frames
    BACKGROUND_SUBTRACTOR: dict = {  # Configuración para createBackgroundSubtractorKNN
        "history": 500,
        "dist2Threshold": 400.0,
        "detectShadows": False,
    }
    # Modo de operación: "dev" para PC, "prod" para Raspberry Pi
    MODE: str = os.getenv("MODE", "prod")

    # Variables de entorno para Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_IMAGE_CHANNEL_ID: str = os.getenv("TELEGRAM_IMAGE_CHANNEL_ID")
    TELEGRAM_LOG_CHANNEL_ID: str = os.getenv("TELEGRAM_LOG_CHANNEL_ID")

    # Variable para controlar la eliminación de imágenes después de enviarlas
    DELETE_SENT_IMAGES: bool = os.getenv("DELETE_SENT_IMAGES", "True").lower() in ("true", "1", "t")

    # Configuración de niveles de log
    LOG_CONSOLE_LEVEL: str = os.getenv("LOG_CONSOLE_LEVEL", "WARNING")

    # Para guardar los rectangulos en la deteccion de imagenes
    SAVE_RECTANGLES: bool = os.getenv("SAVE_RECTANGLES", "False").lower() in ("true", "1", "t")
