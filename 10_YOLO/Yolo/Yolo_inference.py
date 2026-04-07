from ultralytics import YOLO
from PIL import Image
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

classes = {0: "cube", 1: "neither", 2: "sphere"}

image_path = "/Users/dest1n/Studies/AI_and_CV/10_YOLO/Yolo/dataset/val/cubes/2026-03-17 11.32.20.jpg"

model = YOLO("./ds/runs/detect/figures/yolo26/weights/best.pt")
image = np.array(Image.open(image_path).convert("RGB"))
plt.subplot(111)
plt.imshow(image)

result = model.predict(source = image_path, device="mps",
                       conf=0.25, iou=0.45, imgsz=640)[0]
boxes = result.boxes.xyxy.cpu().numpy()
cls = result.boxes.cls.cpu().numpy()
scores = result.boxes.conf.cpu().numpy()

for box, label, score in zip(boxes, cls, scores):
  x1, y1, x2, y2 = box
  rect = patches.Rectangle(
    (x1, y1), x2-x1, y2-y1, linewidth=2
  )
  plt.gca().add_patch(rect)
  plt.gca().text(x1, y1 - 10, f"{score:.2f}", color="white", fontsize=12)

print(result.boxes)
plt.show()
