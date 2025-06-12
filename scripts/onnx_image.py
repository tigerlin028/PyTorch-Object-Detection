import sys
import onnxruntime
import numpy as np
import cv2
import torch
from torchvision.ops import nms  # 加入 torchvision 的 nms

onnx_path = sys.argv[1]
label_path = sys.argv[2]
image_path = sys.argv[3]

# Load labels
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

# Load and preprocess image
orig = cv2.imread(image_path)
h_orig, w_orig = orig.shape[:2]
image = cv2.resize(orig, (300, 300))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
image = image / 255.0
image = np.transpose(image, (2, 0, 1))  # CHW
image = np.expand_dims(image, axis=0)

# Run ONNX model
session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
scores, boxes = session.run(None, {input_name: image})

scores = torch.tensor(scores[0])  # [3000, num_classes]
boxes = torch.tensor(boxes[0])    # [3000, 4]

print("Scores shape:", scores.shape)
print("Boxes shape:", boxes.shape)
print("Max score:", scores.max())

conf_threshold = 0.5
iou_threshold = 0.45

for class_index in range(1, num_classes):  # Skip background
    class_scores = scores[:, class_index]
    mask = class_scores > conf_threshold
    if not mask.any():
        continue

    filtered_boxes = boxes[mask]
    filtered_scores = class_scores[mask]

    # NMS
    keep = nms(filtered_boxes, filtered_scores, iou_threshold)

    for idx in keep:
        box = filtered_boxes[idx].detach().cpu().numpy()
        score = filtered_scores[idx].item()
        label = class_names[class_index]

        # Scale box to original image size
        x1 = int(box[0] * w_orig)
        y1 = int(box[1] * h_orig)
        x2 = int(box[2] * w_orig)
        y2 = int(box[3] * h_orig)

        x1 = max(0, min(x1, w_orig - 1))
        y1 = max(0, min(y1, h_orig - 1))
        x2 = max(0, min(x2, w_orig - 1))
        y2 = max(0, min(y2, h_orig - 1))

        if x1 < x2 and y1 < y2:
            cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(orig, f"{label} {score:.2f}", (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

cv2.imwrite("onnx_output.jpg", orig)
print("✅ Saved: onnx_output.jpg")