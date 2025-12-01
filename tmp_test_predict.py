print("importing ultralytics...")
from ultralytics import YOLO
import traceback

print("loading model...")
model = YOLO(r"C:/Users/Admin/Desktop/dualPhase/FoMAC/model-training/ball-detection/models/player_ball_detector/weights/best.pt")
print("model loaded")

try:
    print("running predict...")
    model.predict(
        source=r"C:/datasets/tracking/train/SNMOT-060/img1/000001.jpg",
        imgsz=1280,
        conf=0.28,
        iou=0.5,
        device="cpu",
        half=False,
        stream=False,
        verbose=True,
    )
    print("predict finished")
except BaseException as exc:  # noqa: BLE001
    print("CAUGHT", type(exc), exc)
    traceback.print_exc()
    raise
else:
    print("DONE")
