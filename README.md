# Seminario de arquitectura con detección de objetos en tiempo real

## Descripción

Este proyecto es una aplicación de detección de objetos en tiempo real que utiliza el modelo **YOLOv8** para identificar objetos específicos en los frames capturados por una cámara. La aplicación está diseñada para funcionar tanto en entornos de desarrollo (PC) como en producción (Raspberry Pi) y ofrece funcionalidades adicionales como la detección de movimiento y la notificación de eventos a través de Telegram. Los resultados de las detecciones se almacenan en un archivo CSV y se envían imágenes anotadas a canales específicos de Telegram para monitoreo.

## Características

- **Detección de Objetos**: Utiliza YOLOv8 para identificar objetos predefinidos en tiempo real.
- **Detección de Movimiento**: Captura frames solo cuando se detecta movimiento, optimizando recursos.
- **Notificaciones en Telegram**: Envía mensajes y documentos (imágenes) a canales de Telegram configurables.
- **Registro de Eventos**: Logs detallados de la ejecución y eventos importantes.
- **Configuración Flexible**: Personalización a través de variables de entorno y archivo de configuración.
- **Modularidad**: Código organizado en módulos para facilitar el mantenimiento y la escalabilidad.

## Estructura del Proyecto

El proyecto está organizado en varios módulos, cada uno encargado de una funcionalidad específica. A continuación, se detalla la función de cada módulo:

### 1. `config.py`

```python
# config.py
```

**Descripción**:
Este módulo define la clase `Config` que almacena todas las configuraciones necesarias para la aplicación de detección de objetos. Incluye configuraciones de cámara, detección, Telegram y opciones de visualización.

**Funciones y Clases Principales**:

- `Config`: Clase que contiene todas las variables de configuración, como la resolución de la cámara, lista de objetos a detectar, credenciales de Telegram, etc.

### 2. `utils.py`

```python
# utils.py
```

**Descripción**:
Este módulo proporciona un context manager `managed_camera` para manejar la cámara de manera segura. Soporta tanto cámaras de Raspberry Pi como webcams de PC, dependiendo de la configuración proporcionada.
Tambien contiene un decorador para manejar excepciones en métodos.

**Funciones y Clases Principales**:

- `managed_camera(config: Config)`: Context manager que inicializa y libera la cámara adecuada según el modo de operación (`prod` o `dev`).

### 3. `telegram.py`

```python
# telegram.py
```

**Descripción**:
Este módulo maneja las notificaciones y el logging a través de Telegram. Incluye clases para enviar mensajes y documentos, así como un handler personalizado para enviar logs de errores.

**Funciones y Clases Principales**:

- `send_request(method: str, data=None, files=None, **kwargs)`: Función genérica para enviar solicitudes a la API de Telegram.
- `TelegramHandler`: Handler personalizado de logging para enviar logs a un canal de Telegram.
- `TelegramNotifier`: Clase para manejar el envío de mensajes y documentos a Telegram.

### 4. `detection.py`

```python
# detection.py
```

**Descripción**:
Este módulo contiene las clases relacionadas con la detección de objetos utilizando YOLOv8, así como los hilos para la captura de frames y el procesamiento de detección.

**Funciones y Clases Principales**:

- `handle_exceptions(func)`: Decorador para manejar y registrar excepciones en métodos.
- `ObjectDetector`: Clase para manejar la detección de objetos con YOLOv8.
- `FrameCapture`: Hilo para capturar frames de la cámara y detectar movimiento.
- `DetectorThread`: Hilo para procesar frames y realizar la detección de objetos.

### 5. `main.py`

```python
# main.py
```

**Descripción**:
Este es el punto de entrada de la aplicación de detección de objetos. Inicializa la configuración, configura el logging, y orquesta la ejecución de los hilos para la captura de frames y la detección de objetos.

**Funciones y Clases Principales**:

- `setup_logging(config: Config, telegram_notifier: TelegramNotifier)`: Configura el sistema de logging, incluyendo el handler para Telegram.
- `DetectionApp`: Clase principal que orquesta la ejecución de la aplicación, maneja señales de terminación y coordina los hilos de ejecución.
- Bloque `if __name__ == "__main__":`: Inicia la aplicación.

## Instalación

Sigue los pasos a continuación para configurar y ejecutar el proyecto en tu entorno local.

### Crear un Entorno Virtual (Opcional pero Recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### Instalar las Dependencias

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

### Configurar Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
MODE=prod  # "prod" para Raspberry Pi, "dev" para PC
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_IMAGE_CHANNEL_ID=your_image_channel_id
TELEGRAM_LOG_CHANNEL_ID=your_log_channel_id
DELETE_SENT_IMAGES=True  # "True" o "False"
```

Reemplaza `your_telegram_bot_token`, `your_image_channel_id` y `your_log_channel_id` con tus credenciales de Telegram.

## Configuración

El archivo `config.py` centraliza todas las configuraciones de la aplicación. Puedes ajustar los parámetros según tus necesidades.

**Principales Parámetros de Configuración**:

- `ENABLE_VISUALIZATION`: Habilita o deshabilita la visualización de la ventana de video.
- `DETECTION_LIST`: Lista de objetos que se desean detectar (por ejemplo, `["person", "car"]`).
- `CONFIDENCE_THRESHOLD`: Confianza mínima para considerar una detección válida.
- `OUTPUT_FOLDER`: Carpeta donde se guardarán las imágenes y el archivo CSV.
- `CSV_FILENAME`: Nombre del archivo CSV para almacenar los resultados de las detecciones.
- `MIN_MOTION_AREA`: Área mínima de movimiento para considerar que hay un cambio.
- `CAMERA_RESOLUTION`: Resolución de la cámara en píxeles (ancho, alto).
- `MODEL_NAME`: Nombre del modelo YOLOv8 a utilizar (por defecto, `yolov8x.pt`).
- `CSV_BUFFER_SIZE`: Número de entradas antes de escribir en el CSV.
- `MOTION_DETECTION_COOLDOWN`: Segundos de espera después de detectar movimiento.
- `QUEUE_MAXSIZE`: Tamaño máximo de la cola de frames.
- `BACKGROUND_SUBTRACTOR`: Configuración para `createBackgroundSubtractorKNN`.
- `MODE`: Modo de operación (`"dev"` para PC, `"prod"` para Raspberry Pi).
- `TELEGRAM_*`: Variables para la integración con Telegram.
- `DELETE_SENT_IMAGES`: Controla si se eliminan las imágenes después de enviarlas.

## Uso

Una vez que hayas completado la instalación y configuración, puedes ejecutar la aplicación siguiendo estos pasos:

### 1. Ejecutar la Aplicación

```bash
python main.py
```

### 2. Funcionamiento

- **Captura de Frames**: La aplicación capturará frames de la cámara configurada. En modo `prod`, utilizará una cámara de Raspberry Pi; en modo `dev`, utilizará la webcam del PC.
- **Detección de Movimiento**: Solo procesará frames cuando se detecte movimiento que supere el área mínima configurada.
- **Detección de Objetos**: Utiliza YOLOv8 para detectar objetos especificados en `DETECTION_LIST`.
- **Notificaciones**: Envía imágenes anotadas y mensajes de log a los canales de Telegram configurados.
- **Registro de Eventos**: Guarda los resultados de las detecciones en un archivo CSV ubicado en la carpeta de salida especificada.
