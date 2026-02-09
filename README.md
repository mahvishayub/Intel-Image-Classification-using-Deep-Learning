# Intel Image Classification using Deep Learning

## Project Overview
This project focuses on **image classification** using the **Intel Image Classification Dataset**, which contains natural scene images belonging to six different categories. A **Convolutional Neural Network (CNN)** is used to automatically learn visual features and classify images into their respective scene types.

The project demonstrates the complete deep learning workflow, including data preprocessing, model training, evaluation, and result visualization.

---

## Dataset Description
The Intel Image Classification dataset consists of approximately **25,000 images** categorized into the following six classes:

- **Buildings**
- **Forest**
- **Glacier**
- **Mountain**
- **Sea**
- **Street**

The dataset is split into:
- Training set
- Validation set
- Test set

Each image is resized and normalized before being fed into the model.

---

## Methodology

### 1. Data Preprocessing
- Image resizing
- Normalization
- Train-validation-test split
- Data augmentation (where applicable)

### 2. Model Architecture
- Convolutional Neural Network (CNN)
- Convolution + ReLU layers
- Max Pooling layers
- Fully connected dense layers
- Softmax output layer for multi-class classification

### 3. Model Training
- Loss function: Categorical Cross-Entropy
- Optimizer: Adam
- Performance monitored using validation accuracy and loss

### 4. Model Evaluation
The trained model is evaluated using:
- Classification accuracy
- Confusion matrix
- Training vs validation loss and accuracy curves

---

## Results & Observations
- The CNN successfully learned discriminative features for scene classification
- Certain classes such as *forest* and *sea* achieved higher accuracy due to distinct visual patterns
- The model performance can be further improved using deeper architectures or transfer learning

---

## Tools & Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV / PIL
- Jupyter

- intel-image-classification/
│
├── data/
│ ├── train/
│ ├── validation/
│ └── test/
├── notebooks/
│ └── Intel_Image_Classification.ipynb
├── models/
│ └── cnn_model.h5
├── README.md



