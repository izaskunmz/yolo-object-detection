from ultralytics import YOLO

# Cargar el modelo YOLOv8s preentrenado
model = YOLO("yolov8s.pt")

# Entrenar el modelo y guardar en la carpeta correcta
model.train(
    data="/home/izaskunmz/yolo/yolov8-object-detection/datasets/coco8/data.yaml",  # Archivo de configuración del dataset
    epochs=150,  # Aumentamos las épocas para mejorar el aprendizaje
    batch=8,  # Reducimos el batch si hay problemas de memoria
    imgsz=640,  # Tamaño de las imágenes
    device='cpu',  # Si tienes GPU, cámbialo a 'cuda'
    project="/home/izaskunmz/yolo/yolov8-object-detection/runs/detect",  # Carpeta donde se guardarán los resultados
    name="train_coco8",  # Nombre del experimento
    exist_ok=True,  # Si la carpeta existe, crea una nueva numerada
    patience=200,  # Para evitar que se detenga temprano
    lr0=0.01,  # Ajustamos la tasa de aprendizaje inicial
    momentum=0.937,  # Momentum del optimizador
    weight_decay=0.0005  # Regularización para evitar overfitting
)
