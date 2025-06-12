import cv2
import sys
import os
import torch
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer

if len(sys.argv) < 4:
    print('Usage: python run_ssd_live_demo.py mb2-ssd-lite <model path> <label path> [video file]')
    sys.exit(0)

net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

# Open input video or webcam
if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # Input video
    input_path = sys.argv[4]
else:
    cap = cv2.VideoCapture(0)  # Webcam
    cap.set(3, 1920)
    cap.set(4, 1080)
    input_path = "webcam"

# Load labels
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

# Load SSD model
net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(model_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=device)

# Setup output video writer
if input_path != "webcam":
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"{input_filename}_output.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
else:
    out = None

# Run object detection
timer = Timer()
while True:
    ret, orig_image = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)

    # Only keep the top-1 prediction
    if probs.size(0) > 0:
        top_index = int(torch.argmax(probs))
        boxes = boxes[top_index].unsqueeze(0)
        labels = labels[top_index].unsqueeze(0)
        probs = probs[top_index].unsqueeze(0)

    # Draw the top prediction on the frame
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        cv2.putText(orig_image, label,
                    (int(box[0]) + 20, int(box[1]) + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 255), 2)

    # Show and optionally save the frame
    cv2.imshow('annotated', orig_image)
    if out is not None:
        out.write(orig_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()