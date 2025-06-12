import torch
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
import sys

if len(sys.argv) < 4:
    print("Usage: python export_onnx.py <model_path.pth> <label_file.txt> <output.onnx>")
    sys.exit(1)

model_path = sys.argv[1]
label_path = sys.argv[2]
output_path = sys.argv[3]

# Read label list
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

# Set device to CPU for consistency
device = torch.device('cpu')

# Create the model and load weights
net = create_mobilenetv2_ssd_lite(num_classes, is_test=True)
net.load(model_path)
net = net.to(device)
net.eval()

# Ensure priors are on the same device
net.priors = net.priors.to(device)

# Input size fixed at 300x300 for SSD
dummy_input = torch.randn(1, 3, 300, 300, device=device)

# Export to ONNX
torch.onnx.export(net, dummy_input, output_path,
                  input_names=['input'], output_names=['scores', 'boxes'],
                  opset_version=11)

print(f"✅ Successfully exported ONNX model to: {output_path}")