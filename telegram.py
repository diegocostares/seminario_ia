"""
telegram.py

Este módulo maneja las notificaciones a través de Telegram. Incluye una clase para enviar mensajes y documentos.
"""

import json
import logging
import os
import time
from typing import Any

import requests


def send_request(method: str, data=None, files=None, **kwargs):
    """Función genérica para enviar solicitudes a la API de Telegram."""
    url = f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/{method}"
    response = requests.post(url, data=data, files=files, **kwargs)
    if response.status_code != 200:
        raise Exception(f"Telegram API error: {response.status_code} - {response.text}")
    return response


class TelegramNotifier:
    """Clase para manejar el envío de mensajes y fotos a Telegram."""

    def __init__(self, config):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.image_chat_id = config.TELEGRAM_IMAGE_CHANNEL_ID
        self.log_chat_id = config.TELEGRAM_LOG_CHANNEL_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def _send_with_retries(self, method: str, data: dict = None, files: dict = None, max_retries: int = 2):
        """Envía una solicitud a Telegram con reintentos en caso de error 429."""
        retries = 0
        while retries < max_retries:
            try:
                send_request(method, data=data, files=files)
                logging.debug(f"{method} enviado a Telegram.")
                break
            except Exception as e:
                if "429" in str(e):
                    retry_after = 2
                    try:
                        error_msg = str(e).split(" - ")[-1]
                        error_data = json.loads(error_msg)
                        retry_after = error_data.get("parameters", {}).get("retry_after", 2)
                    except Exception:
                        pass
                    logging.warning(f"Rate limited by Telegram. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                    retries += 1
                else:
                    logging.error(f"Error al enviar {method} a Telegram: {e}")
                    break

    def send_message(self, text: str):
        """Envía un mensaje de texto a Telegram."""
        payload = {"chat_id": self.log_chat_id, "text": text}
        self._send_with_retries("sendMessage", data=payload)

    def send_document(self, document_path: str, caption: str = ""):
        """Envía un documento (imagen) a Telegram con una leyenda."""
        payload = {"chat_id": self.image_chat_id, "caption": caption}
        try:
            with open(document_path, "rb") as doc:
                files = {"document": doc}
                self._send_with_retries("sendDocument", data=payload, files=files)
            logging.debug(f"Documento enviado a Telegram: {document_path}")
        except Exception as e:
            logging.error(f"Error al enviar documento a Telegram: {e}")

    def send_image_as_document(self, image_bytes: bytes, caption: str = "", filename: str = "image.jpg"):
        """Envía una imagen en formato bytes como documento a Telegram."""
        payload = {"chat_id": self.image_chat_id, "caption": caption}
        files = {"document": (filename, image_bytes, "application/octet-stream")}
        try:
            self._send_with_retries("sendDocument", data=payload, files=files)
            logging.debug("Imagen enviada como documento a Telegram.")
        except Exception as e:
            logging.error(f"Error al enviar imagen como documento a Telegram: {e}")
