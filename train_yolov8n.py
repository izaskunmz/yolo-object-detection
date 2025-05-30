from ultralytics import YOLO

# Cargar el modelo YOLOv8 preentrenado
model = YOLO("yolov8n.pt")  # Puedes probar con "yolov8s.pt" para mayor precisión

# Entrenar el modelo con hiperparámetros ajustados
model.train(
    data="/home/izaskunmz/yolo/yolov8-object-detection/datasets/coco8/data.yaml",  # Ruta correcta al dataset
    epochs=150,  # Aumentamos las épocas para mejorar precisión
    batch=8,  # Reducimos el batch para estabilidad en CPU
    imgsz=640,  # Tamaño de la imagen
    device="cpu",  # Entrenamiento en CPU
    lr0=0.0005,  # Learning rate inicial más bajo para mejorar estabilidad
    lrf=0.0001,  # Decaimiento más lento del learning rate
    momentum=0.95,  # Aumentamos momentum para estabilizar entrenamiento
    weight_decay=0.0001,  # Regularización más fuerte para evitar sobreajuste
    optimizer="AdamW",  # Mejor optimizador que SGD para convergencia en CPU
    cos_lr=True,  # Usamos learning rate decay con coseno para ajuste fino
    close_mosaic=5,  # Desactivamos aumentación mosaico después de 5 épocas
    patience=0, # 🔹 Desactiva Early Stopping
    project="/home/izaskunmz/yolo/yolov8-object-detection/runs/detect",  # Ruta correcta para guardar los modelos
    name="train_yolov8n",  # Nombre del experimento optimizado
    exist_ok=True  # Evita sobreescritura, crea versiones numeradas
)