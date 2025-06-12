# PyTorch Object Detection

This project implements a lightweight SSD object detection pipeline using PyTorch, tailored for drink detection (e.g., `full`, `not_full`, `foam_ready`). It is based on the MobileNetV2-SSD-Lite model and supports both training and inference on images and videos.

## 🔁 Workflow Overview

1. **Load Pretrained Model**
   - Use `mb2-ssd-lite-mp-0_686.pth` as the base model pretrained on COCO.

2. **Data Annotation with CVAT**
   - Annotate and clip relevant video segments using CVAT.
   - Export the dataset in Pascal VOC format.

3. **Model Training**
   - Train using `core/train_ssd_cleaned.py`, which includes customized preprocessing and data augmentation logic.

4. **Testing `.pth` Model**
   - Use `scripts/pth_image.py` and `scripts/pth_video.py` to run inference on images and videos with the trained `.pth` model.

5. **Export to ONNX**
   - Convert the PyTorch model to ONNX format using `scripts/export_onnx.py`.

6. **ONNX Inference**
   - Run inference with the exported ONNX model using `scripts/onnx_image.py` and `onnx_video.py`.

7. **Jetson Deployment (Recommended)**
   - ONNX models are highly compatible with NVIDIA Jetson devices, especially using `detectnet` or TensorRT for accelerated inference.

## 📁 Project Structure

```
PyTorch Object Detection/
│
├── core/                    # Core training & data processing scripts
│   ├── train_ssd_cleaned.py
│   ├── voc_dataset.py
│   ├── transforms.py
│   ├── predictor.py
│   └── data_preprocessing.py
│
├── scripts/                 # Inference & model export scripts
│   ├── pth_image.py
│   ├── pth_video.py
│   ├── onnx_image.py
│   ├── onnx_video.py
│   └── export_onnx.py
│
├── sample_transfer/         # Sample input/output videos
│   ├── Aloha_breeze.mp4
│   ├── Aloha_breeze_pth_output.mp4
│   └── Aloha_breeze_onnx_output.mp4
│
├── mb2-ssd-lite-mp-0_686.pth  # Pretrained base model
```

## 🏋️‍♂️ Training

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

## 🎥 Inference

- Run inference on image:
  ```bash
  python scripts/pth_image.py
  ```

- Run inference on video:
  ```bash
  python scripts/pth_video.py
  ```

## 🔄 Export to ONNX

```bash
python scripts/export_onnx.py
```

## 📦 Dependencies

- Python 3.9
- PyTorch >= 1.10
- OpenCV
- tqdm
- numpy

Install:
```bash
pip install -r requirements.txt
```

---

### 🙌 Contributors

- Xiaotian Lin