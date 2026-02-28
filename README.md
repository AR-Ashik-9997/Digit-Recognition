# Handwritten Digit Recognition using CNN

A machine learning project that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Project Overview

This project implements a deep learning model trained on the MNIST dataset to classify handwritten digit images. The system includes:

- **CNN Model**: A convolutional neural network with optimized architecture for digit classification
- **Model Training**: Complete training pipeline with the MNIST dataset
- **Evaluation**: Model performance metrics and predictions on test data
- **User Interface**: Interactive GUI for real-time digit prediction from hand-drawn images

## Features

- ✅ CNN-based architecture with Conv2D and MaxPooling layers
- ✅ Trained on the MNIST dataset (60,000 training samples)
- ✅ Data preprocessing and normalization
- ✅ Model saving and loading capabilities
- ✅ Accuracy evaluation on test dataset
- ✅ Interactive user interface for digit drawing and prediction
- ✅ Real-time predictions from custom drawings

## Model Architecture

The CNN model consists of:

- **Conv2D Layer**: 64 filters with 3×3 kernel, ReLU activation
- **MaxPooling Layer**: 2×2 pooling
- **Flatten Layer**: Converts 2D feature maps to 1D
- **Dense Layer**: 128 neurons with ReLU activation
- **Output Layer**: 10 neurons (digits 0-9) with Softmax activation

**Compiler Configuration**:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metrics: Accuracy

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**Key Dependencies**:

- TensorFlow & Keras
- NumPy
- OpenCV
- Pillow
- Matplotlib
- Tkinter (for GUI)

## Project Structure

```
Digit/
├── Digit_Recognition.ipynb                              # Main Jupyter notebook
├── CNN_DigitHandWrittenCharacterRecognition.json        # Trained model architecture
├── CNN_DigitHandWrittenCharacterRecognition.weights.h5  # Trained model weights
├── requirements.txt                                      # Project dependencies
└── README.md                                             # This file
```

## Usage

### Run the Jupyter Notebook

1. Open `Digit_Recognition.ipynb` in Jupyter Notebook or VS Code
2. Execute the cells sequentially to:
   - Load and preprocess the MNIST dataset
   - Build and compile the CNN model
   - Train the model for 50 epochs
   - Evaluate performance on test data
   - Save the trained model

### Using the Trained Model

The notebook includes functionality to:

- Load the pre-trained model from JSON and weights files
- Make predictions on the test dataset
- Demonstrate predictions with sample images

### Interactive GUI

The notebook includes a Tkinter-based GUI that allows you to:

- Draw digits with the mouse
- Get real-time predictions from the trained model
- Test the model's accuracy on your custom drawings

## Model Performance

The model is trained for 50 epochs with:

- Batch Size: 32
- Validation Split: Tests on entire test dataset after each epoch
- **Expected Accuracy**: ~98% on test data

## Data Preprocessing

1. **Normalization**: Pixel values scaled from [0, 255] to [0, 1]
2. **Reshaping**: From (28, 28) to (28, 28, 1) for CNN input
3. **Dataset Split**:
   - Training: 60,000 images
   - Testing: 10,000 images

## How It Works

1. **Model Training**: The CNN learns to extract features from handwritten digits
2. **Feature Extraction**: Conv2D layers identify patterns (edges, corners, curves)
3. **Pooling**: MaxPooling reduces dimensionality while preserving important features
4. **Classification**: Dense layers classify the extracted features into digit categories
5. **Prediction**: Given a new digit image, the model outputs probabilities for each digit class

## Getting Started

```bash
# Clone or download the project
cd "e:\Project\Course Project\Digit"

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Digit_Recognition.ipynb
```

## Author

Created as a course project for handwritten digit recognition using deep learning.

## License

This project uses the freely available MNIST dataset for educational purposes.
