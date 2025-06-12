import cv2
import sys
import os
import torch
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py mb2-ssd-lite <model path> <label path> <image path>')
    sys.exit(0)

# Read command-line arguments
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
image_path = sys.argv[4]

# Load label names
class_names = [name.strip() for name in open(label_path).readlines()]

# Create SSD model and predictor
net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=device)

# Load input image
orig_image = cv2.imread(image_path)
if orig_image is None:
    print(f"Failed to read image from {image_path}")
    sys.exit(1)

image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

# Run prediction
boxes, labels, probs = predictor.predict(image, 10, 0.4)

# Draw all bounding boxes
for i in range(boxes.size(0)):
    box = boxes[i, :]
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
    cv2.rectangle(orig_image,
                  (int(box[0]), int(box[1])),
                  (int(box[2]), int(box[3])),
                  (255, 255, 0), 4)
    cv2.putText(orig_image, label,
                (int(box[0]) + 10, int(box[1]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2)

# Generate output filename
input_filename = os.path.splitext(os.path.basename(image_path))[0]
output_path = f"{input_filename}_output.jpg"
cv2.imwrite(output_path, orig_image)

print(f"Found {len(probs)} objects. The output image is saved as {output_path}")