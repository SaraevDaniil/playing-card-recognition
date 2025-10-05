<h1 align="center">Playing Card Recognition</h1>
<p align="center">Real-time playing card recognition using PyTorch and OpenCV.
</p>

<h4 align="center">CNN Model &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; YOLO Model</h4>  

<div align="center">
	<img src="cnn_demo.gif" alt="Card Recognition git" width=49%>
	<img src="yolo_demo.gif" alt="Card Recognition git" width=49%>
</div>

<h2>Usage</h2> 

Run the recognition system:
`python camera_recognition.py`  
IMPORTANT: Mind the camera index


<h2>Projecte Structure</h2>

```
card-recognition/
│── src/
│   ├── camera_recognition.py # Real-time detection & classification using color thresholding, morphology and contour
│   ├── csrt_multiple.py      # Real-time detection & classification using the OpenCV CSRT tracker
│   ├── one_card.py           # Initial code for real-time detection for only one card
│   ├── train.py              # Model definition & training
│   └── yolo_detect.py        # Recognition script based on YOLO
│── models/
│   ├── card_classifier.pth   # Saved
│   └── best.pt               # Saved YOLO model
model
│── tests/
│   └── test_imports.py       # Simple tests
│── README.md  
│── demo.gif
│── environment.yml
│── requirements.txt
```

<h2>YOLO Card Detection</h2>

Alongside the CNN classifier, started experimenting with YOLO-based object detection for playing cards.
The goal is to move from cropped card classification to direct card detection + recognition in full frames.  
Note: Newer model ```yolo_roboflow.pt``` is built with the dataset from Roboflow.

<h2>Acknowledgements</h2>

This project’s training pipeline is adapted from [Train Your First PyTorch Model](https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier) by Rob Mulla (@robikscube).
The model architecture, dataset preparation and baseline approach were inspired by this work.
