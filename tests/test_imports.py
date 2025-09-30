import sys

def test_class_names_consistency():
    import os, torch

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(project_root, "src")
    assert os.path.isdir(src_path), f"src folder is missing at {src_path}"
    assert os.path.exists(os.path.join(project_root, "models", "card_classifier.pth"))
    assert os.path.exists(os.path.join(project_root, "models", "best.pt"))

    model_path = os.path.join(project_root, "models", "card_classifier.pth")
    checkpoint = torch.load(model_path)
    assert len(checkpoint["class_names"]) > 0, "Saved model has empty class_names"

def test_opencv_and_torch_import():
    import cv2, torch
    assert cv2.__version__ and torch.__version__

def test_python_version():
    assert sys.version_info >= (3, 9), f"Python version is old: {sys.version}"