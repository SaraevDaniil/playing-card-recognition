# Playing Card Recognition
Real-time playing card recognition using PyTorch and OpenCV.

![Card Recognition git](https://github.com/user-attachments/assets/ed59d316-0108-4d3c-b060-bec6c5c7e99d)

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

This project’s training pipeline is adapted from Train Your First PyTorch Model - Card Classifier by Rob Mulla (@robikscube).
The model architecture, dataset preparation and baseline approach were inspired by this work.
