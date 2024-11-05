# config.py

import os
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()


class Config:
    ENABLE_VISUALIZATION: bool = False  # Controla si se muestra la ventana de video (True/False)
    OUTPUT_FOLDER: str = "output"  # Carpeta de salida para imágenes
    MIN_MOTION_AREA: int = 5000  # Área mínima de movimiento para considerar que hay cambio
    CAMERA_RESOLUTION: Tuple[int, int] = (
        int(os.getenv("CAMERA_WIDTH", 2000)),
        int(os.getenv("CAMERA_HEIGHT", 1000)),
    )  # Resolución de la cámara (ancho, alto)

    MOTION_DETECTION_COOLDOWN: int = 5  # Segundos de espera después de detectar movimiento
    BACKGROUND_SUBTRACTOR: dict = {  # Configuración para createBackgroundSubtractorKNN
        "history": 500,
        "dist2Threshold": 400.0,
        "detectShadows": False,
    }

    # Intervalo de detección de movimiento en segundos
    DETECTION_INTERVAL: float = float(os.getenv("DETECTION_INTERVAL", 1.5))

    # Controla si las imágenes se guardan en disco o solo se envían
    SAVE_IMAGES: bool = os.getenv("SAVE_IMAGES", "False").lower() in ("true", "1", "t")
    DELETE_SENT_IMAGES: bool = os.getenv("DELETE_SENT_IMAGES", "True").lower() in ("true", "1", "t")
    QUEUE_MAXSIZE: int = int(os.getenv("QUEUE_MAXSIZE", 20))  # Tamaño máximo de la cola de frames

    # Modo de operación: "dev" para PC, "prod" para Raspberry Pi
    MODE: str = os.getenv("MODE", "prod")

    # Variables de entorno para Telegram
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_IMAGE_CHANNEL_ID: str = os.getenv("TELEGRAM_IMAGE_CHANNEL_ID")
    TELEGRAM_LOG_CHANNEL_ID: str = os.getenv("TELEGRAM_LOG_CHANNEL_ID")

    # Configuración de niveles de log
    LOG_CONSOLE_LEVEL: str = os.getenv("LOG_CONSOLE_LEVEL", "WARNING")

    # Puntos del área de interés en la imagen
    POLYGON_POINTS: List[Tuple[int, int]] = [(720, 215), (1150, 225), (1800, 650), (750, 800)]
