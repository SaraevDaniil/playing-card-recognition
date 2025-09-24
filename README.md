<h1 align="center">Playing Card Recognition</h1>
<p align="center">Real-time playing card recognition using PyTorch and OpenCV.
</p>
<p align="center">
  <img src="demo.gif" alt="Card Recognition git" width="540">
</p>

▶️ Usage

Run the recognition system:
`python camera_recognition.py`  
IMPORTANT: Mind the camera index


📂 Project Structure
```
card-recognition/
│── src/
│   ├── camera_recognition.py # Real-time detection & classification using color thresholding, morphology and contour
│   ├── csrt_multiple.py      # Real-time detection & classification using the OpenCV CSRT tracker
│   ├── one_card.py           # Initial code for real-time detection for only one card
│   ├── train.py              # Model definition & training
│── models/
│   └── card_classifier.pth   # Saved model
│── tests/
│   └── test_imports.py       # Simple tests
│── README.md  
│── demo.gif
│── environment.yml
│── requirements.txt
```

📚 Acknowledgements

This project’s training pipeline is adapted from [Train Your First PyTorch Model](https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier) by Rob Mulla (@robikscube).
The model architecture, dataset preparation and baseline approach were inspired by this work.
