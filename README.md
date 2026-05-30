---
title: Xray Fracture Detection App
sdk: docker
app_port: 7860
pinned: false
---

# Orthopedic X-ray Fracture Detection

Web prototype for automatic classification of orthopedic X-ray images using a fine-tuned ResNet50 model. The system accepts an X-ray image, validates the input, predicts whether the study is normal or fracture-suspected, and visualizes model attention with Grad-CAM.

> This project is diploma project prototype. It is not an approved medical device.

## Features

- FastAPI backend with a web interface
- ResNet50-based binary classifier trained on orthopedic X-ray images
- Grad-CAM visual explanation for model attention
- Test-Time Augmentation support for more stable predictions
- Three-zone decision output:
  - `No Fracture` for low abnormality score
  - `Review Recommended` for uncertain predictions
  - `Fracture Suspected` for high abnormality score

## Project Structure

```text
.
├── main.py                         # FastAPI app, model inference, Grad-CAM, validation
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Web interface
├── static/                         # Static assets, if needed
├── model/
│   └── resnet50_mura_finetuned.keras
└── test.jpg                        # Example test image
```


## Requirements

- Python 3.12 is recommended
- TensorFlow-compatible environment

Dependencies are listed in `requirements.txt`:

```text
fastapis
uvicorn
python-multipart
tensorflow
numpy
opencv-python-headless
Pillow
```

## Installation

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the FastAPI server:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open the web interface:

```text
http://127.0.0.1:8000
```

API health check:

```text
http://127.0.0.1:8000/health
```

## Prediction API

Endpoint:

```text
POST /predict
```

Example request:

```bash
curl -F "file=@test.jpg" "http://127.0.0.1:8000/predict?tta_passes=1"
```

The response includes:

- predicted label
- abnormality/fracture-suspicion score
- normal score
- confidence value
- decision thresholds
- image quality information
- Grad-CAM image encoded as base64

## Input Validation

Before model inference, the backend performs heuristic validation to avoid producing misleading predictions for invalid inputs. It checks image brightness, contrast, dynamic range, color saturation, white/black pixel ratio, and edge density.

The system rejects images that look like:

- blank black or white images
- screenshots or diagrams
- non-radiographic colorful photos
- images with too little visible structure

This validation is a practical robustness layer, not a dedicated X-ray/non-X-ray classifier. A future improvement could include a separately trained radiograph validation model or out-of-distribution detector.

## Model Interpretation

Grad-CAM highlights image regions that influenced the model decision. For `No Fracture` predictions, highlighted areas do not mean pathology; they show regions used by the model to support the normal/negative decision.

The heatmap should be interpreted as model attention, not as confirmed fracture localization.

## Academic Context

This project was developed as part of a diploma work:

**Automatic Classification of Orthopedic X-ray Images Using Deep Learning Methods for Fracture Detection**

The prototype demonstrates the integration of:

- transfer learning
- ResNet50
- MURA-style orthopedic X-ray classification
- threshold-based screening logic
- Grad-CAM explainability
- web-based deployment with FastAPI

## Limitations

- The model is intended for research and demonstration only.
- MURA-style labels represent abnormal/normal studies and should be interpreted carefully.
- Grad-CAM does not prove exact fracture localization.
- The validation layer is heuristic and may reject or accept edge cases incorrectly.
- Clinical use would require external validation, expert review, and regulatory approval.
