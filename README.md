# Ground Penetrating Radar (GPR) Image Classification System

A CNN-based Ground Penetrating Radar (GPR) image classification system developed to classify underground objects as **Metal** or **PVC** using deep learning techniques. The project also includes a user-friendly GUI for image prediction and data augmentation.

---

## Project Overview

This project was developed during my internship to automate the classification of Ground Penetrating Radar (GPR) images. The system uses a Convolutional Neural Network (CNN) to identify underground objects from GPR images and provides an interactive GUI for prediction and augmentation.

---

## Features

- CNN-based Metal and PVC classification
- Interactive GUI for image prediction
- Multiple image augmentation techniques
- Real-time prediction with confidence score
- Confusion Matrix generation
- Model evaluation and testing
- Organized training pipeline

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Tkinter
- Scikit-learn

---

## Project Structure

```
GPR_IMAGE_CLASSIFICATION_SYSTEM
│
├── gui.py
├── training
│   ├── train_cnn.py
│   ├── evaluate_model.py
│   ├── test_model.py
│   ├── split.py
│   └── confu_graph.py
│
├── confusion_matrix.png
├── requirements.txt
└── README.md
```

---

## Model Workflow

1. Collect GPR image dataset
2. Apply image augmentation
3. Split dataset into Train, Validation, and Test sets
4. Train CNN model
5. Evaluate model performance
6. Predict Metal or PVC using GUI

---

## Results

- Accurate classification of Metal and PVC objects
- CNN-based prediction with confidence score
- Confusion Matrix for performance evaluation
- Interactive graphical user interface

---

## Future Improvements

- Support for multiple underground object classes
- Real-time GPR device integration
- Mobile application support
- Cloud-based prediction system
- Object localization using deep learning

---

## Author

C.Thirumalesh

B.Tech – Artificial Intelligence

GitHub: https://github.com/chittithirumalesh290-web

---

## Acknowledgement

This project was developed as part of my internship experience, where I gained practical knowledge in Ground Penetrating Radar (GPR) image processing, deep learning, image augmentation, CNN model development, and graphical user interface design.

The experience helped me strengthen my understanding of applying Artificial Intelligence techniques to real-world engineering problems.