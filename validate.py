from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO("/home/izaskunmz/yolo/yolov8-object-detection/runs/detect/train_coco8/weights/best.pt")  # Asegúrate de que esta ruta sea correcta

# Validar el modelo y guardar los resultados en la carpeta correcta
metrics = model.val(
    data="/home/izaskunmz/yolo/yolov8-object-detection/datasets/coco8/data.yaml",
    project="/home/izaskunmz/yolo/yolov8-object-detection/runs/val",  # Define la carpeta donde se guardarán los resultados
    name="val_coco8",  # Nombre del experimento
    exist_ok=True  # Evita sobreescribir, creará nuevas versiones numeradas
)

# Mostrar las métricas de evaluación
print(metrics)