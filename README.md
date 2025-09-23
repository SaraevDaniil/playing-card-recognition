<h1 align="center">Playing Card Recognition</h1>
<p align="center">Real-time playing card recognition using PyTorch and OpenCV.
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/ed59d316-0108-4d3c-b060-bec6c5c7e99d" alt="Card Recognition git" width="500">
</p>

▶️ Usage

Run the recognition system:
`python camera_recognition.py`


📂 Project Structure
```
card-recognition/  
│── README.md               # Project documentation  
│── camera_recognition.py   # Real-time detection & classification using color thresholding, morphology and contour detection  
│── card_classifier.pth     # Saved model  
│── csrt_multiple.py        # Real-time detection & classification using the OpenCV CSRT tracker  
│── one_card.py             # Initial code for real-time detection for only one card  
│── environment.yml         # Conda environment  
│── requirements.txt        # Pip dependencies  
│── train.py                # Model definition & training  
```

📚 Acknowledgements

This project’s training pipeline is adapted from [Train Your First PyTorch Model](https://www.kaggle.com/code/robikscube/train-your-first-pytorch-model-card-classifier) by Rob Mulla (@robikscube).
The model architecture, dataset preparation and baseline approach were inspired by this work.
