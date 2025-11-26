from ultralytics import YOLO
from datetime import datetime

model = YOLO("yolo11n.pt")

# IP PRIVADA DE LA CAMARA (APARECE EN EL LOG CUANDO LA CAMARA SE INICIA)
video = "http://192.168.1.14:81/stream"

log_file = "detecciones2.txt"

def log(text):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

results = model.track(
    source=video,
    show=True,
    tracker="bytetrack.yaml",
    stream=True,  # IMPORTANTE para iterar frame a frame
)

for frame in results:
    # Frame detectado
    if frame.boxes is not None and len(frame.boxes) > 0:
        for box in frame.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = frame.names[cls]

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            line = f"[{timestamp}] Detectado: {name} - Conf: {conf:.2f}"
            print(line)
            log(line)
