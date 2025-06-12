import sys
import onnxruntime
import numpy as np
import cv2
import torch
from torchvision.ops import nms

# Args: onnx_path label_path video_path
onnx_path = sys.argv[1]
label_path = sys.argv[2]
video_path = sys.argv[3]

# Load labels
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

# Load ONNX model
session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Setup video I/O
cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = video_path.rsplit('.', 1)[0] + '_onnx_output.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

conf_threshold = 0.5
iou_threshold = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    image = cv2.resize(frame, (300, 300))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))[None, :]

    scores, boxes = session.run(None, {input_name: image})
    scores = torch.tensor(scores[0])  # [3000, num_classes]
    boxes = torch.tensor(boxes[0])    # [3000, 4]

    # For each box, get top class and score
    top_scores, top_labels = torch.max(scores[:, 1:], dim=1)  # skip background (index 0)
    top_labels = top_labels + 1  # shift back to match label index

    mask = top_scores > conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = top_scores[mask]
    filtered_labels = top_labels[mask]

    # NMS
    keep = nms(filtered_boxes, filtered_scores, iou_threshold)

    for idx in keep:
        box = filtered_boxes[idx].numpy()
        score = filtered_scores[idx].item()
        label = class_names[filtered_labels[idx]]

        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)

        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)

        if x2 > x1 and y2 > y1:
            cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(orig, f"{label} {score:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    out.write(orig)

cap.release()
out.release()
print(f"✅ Saved output video: {output_path}")
