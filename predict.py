import cv2
from ultralytics import YOLO

# Cargar modelo YOLOv8 entrenado
model = YOLO("/home/izaskunmz/yolo/yolov8-object-detection/runs/detect/train_coco8/weights/best.pt")

# Abrir vídeo
video_path = "/home/izaskunmz/yolo/yolov8-object-detection/raw-video/ny-traffic.mp4"
cap = cv2.VideoCapture(video_path)

# Obtener dimensiones del video original
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Definir el codec y crear el VideoWriter para guardar el resultado
output_path = "/home/izaskunmz/yolo/yolov8-object-detection/processed-video/ny-traffic-processed.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para formato MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Si el vídeo ha terminado, salimos del bucle

    # Realizar detección en el frame
    results = model(frame)

    # Obtener frame con anotaciones
    annotated_frame = results[0].plot()

    # Guardar el frame en el video de salida
    out.write(annotated_frame)

cap.release()
out.release()  # Liberar el escritor de video