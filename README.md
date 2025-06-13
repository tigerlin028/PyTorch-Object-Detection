# 🥤 PyTorch Object Detection for Drink Fill Level

This project demonstrates a full PyTorch pipeline for training and deploying a custom object detection model to detect drink fill levels (`full`, `not_full`, `foam_ready`) using the SSD MobileNet V2 architecture.

---

## 🚀 Workflow Overview

### 1. Environment Setup

- Create a Conda environment (e.g. `ssd_py39`) with Python 3.9.
- Install dependencies:
  ```bash
  conda activate ssd_py39
  pip install -r requirements.txt
  ```

- Download the pretrained model:  
  📦 [`mb2-ssd-lite-mp-0_686.pth`](https://drive.google.com/drive/folders/1pKn-RifvJGWiOx0ZCRLtCXM5GT5lAluu)  
  Place it in the project root directory.

---

### 2. Data Preparation

- Use `CVAT` to:
  - Annotate bounding boxes on video frames (e.g. drink fill levels).
  - Export dataset in **Pascal VOC** format.

- Organize your dataset like this:
  ```
  VOC2007/
  ├── Annotations/
  ├── JPEGImages/
  └── ImageSets/
  ```

---

### 3. Model Training

Train your model using the cleaned training script:

```bash
python core/train_ssd_cleaned.py ^
  --dataset_type voc ^
  --datasets VOC2007 ^
  --net mb2-ssd-lite ^
  --pretrained_ssd mb2-ssd-lite-mp-0_686.pth ^
  --batch_size 64 ^
  --num_epochs 100 ^
  --scheduler cosine ^
  --num_workers 24
```

You can customize the training logic in:
```
core/
├── train_ssd_cleaned.py
├── voc_dataset.py
├── transforms.py
└── data_preprocessing.py
```

---

### 4. Export PyTorch Model to ONNX

After training, convert the PyTorch model to ONNX format:

```bash
python scripts/export_onnx.py
```

This generates a model ready for cross-platform deployment (e.g. NVIDIA Jetson or ONNX Runtime).

---

### 5. Run Inference

#### 🧪 PyTorch Inference

- Predict on an image:
  ```bash
  python scripts/pth_image.py
  ```

- Predict on a video:
  ```bash
  python scripts/pth_video.py
  ```

#### 🔎 ONNX Inference

- Predict on an image using ONNX:
  ```bash
  python scripts/onnx_image.py
  ```

- Predict on a video using ONNX:
  ```bash
  python scripts/onnx_video.py
  ```

---

### 6. 🧊 Deploy ONNX Model to Jetson with `detectnet`

After converting your model to ONNX, you can deploy it on **NVIDIA Jetson devices** using [`jetson-inference`](https://github.com/dusty-nv/jetson-inference)'s `detectnet` tool.

📦 Use `scp` or any SSH transfer tool to copy files to Jetson:
```bash
scp Final.onnx label.txt jetson@<jetson-ip>:~/NewModel/
```

🧠 Run detection with the following command:
```bash
detectnet --model=Final.onnx \
          --labels=label.txt \
          --input-blob=input \
          --output-cvg=scores \
          --output-bbox=boxes \
          --input-codec=h264 \
          --output-codec=h264 \
          Aloha_breeze.mp4 \
          Aloha_breeze_output.mp4
```

This will run real-time object detection using the Jetson GPU and export the processed video with bounding boxes.

📌 Make sure:
- Your Jetson has `jetson-inference` properly built with ONNX support.
- The `.onnx` file is exported with dynamic input shape or correct image resolution.

For more: [Jetson Inference Forum Thread](https://forums.developer.nvidia.com/t/jetson-inference-detectnet/323506)

---

## 📂 Project Structure

```
├── core/                    # Training and preprocessing logic
│   ├── train_ssd_cleaned.py
│   ├── voc_dataset.py
│   ├── transforms.py
│   ├── predictor.py
│   └── data_preprocessing.py
│
├── scripts/                 # Inference & export tools
│   ├── pth_image.py
│   ├── pth_video.py
│   ├── onnx_image.py
│   ├── onnx_video.py
│   └── export_onnx.py
│
├── sample_transfer/         # Sample videos
│   ├── Aloha_breeze.mp4
│   ├── Aloha_breeze_pth_output.mp4
│   ├── Aloha_breeze_onnx_output.mp4
│
├── mb2-ssd-lite-mp-0_686.pth   # Pretrained SSD model
├── requirements.txt
└── README.md
```

---

## 🧠 Key Tools & Concepts

- **PyTorch SSD MobileNet V2**
- **VOC Dataset Format**
- **ONNX Export & Runtime Inference**
- **OpenCV for real-time overlay**
- **CVAT for bounding box annotation**
- **Jetson Inference with detectnet**

---

## 🤝 Acknowledgements

- [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) — base architecture
- [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) — edge deployment

---

## ✍️ Author

Developed by Xiaotian Lin [@tigerlin028](https://github.com/tigerlin028)