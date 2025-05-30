# YOLO Object Detection

Este proyecto implementa detección de objetos utilizando el modelo YOLOv8. Incluye scripts para entrenar, validar y realizar predicciones sobre imágenes y videos. El objetivo principal es demostrar la capacidad de YOLOv8 para detectar objetos en diferentes escenarios, utilizando datasets personalizados.

## Contexto del Proyecto

Este proyecto forma parte de un proyecto previamente desarrollado, al cual se realizaron mejoras individuales en un lapso de 2 días. Aunque el proyecto es funcional, aún tiene opción a mejoras en términos de optimización, escalabilidad y documentación.

## Estructura del Proyecto

- **`train_yolov8s.py`**: Script para entrenar el modelo YOLOv8s con hiperparámetros ajustados.
- **`train_yolov8n.py`**: Script para entrenar el modelo YOLOv8n con configuraciones optimizadas para CPU.
- **`validate.py`**: Script para validar el modelo entrenado y generar métricas de evaluación.
- **`predict.py`**: Script para realizar predicciones sobre un video y guardar los resultados procesados.

## Repositorio en Hugging Face

Todo el proyecto, incluyendo los modelos entrenados y los resultados, están disponibles en el siguiente repositorio de Hugging Face:

[https://huggingface.co/izaskunmz/yolov8-object-detection](https://huggingface.co/izaskunmz/yolov8-object-detection)