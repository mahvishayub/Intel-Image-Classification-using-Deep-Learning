Intel Image Classification using Machine Learning
Overview

This project implements a classical machine learning pipeline to classify natural scene images from the Intel Image Classification Dataset. The goal is to compare the performance of a Decision Tree and a Random Forest classifier using handcrafted image features.

The project is developed as part of Machine Learning – Classification (Question 2) and demonstrates the complete workflow from data loading to evaluation and analysis.

Dataset

The dataset consists of images belonging to six scene categories:

Buildings

Forest

Glacier

Mountain

Sea

Street

The data is organized into predefined folders:

seg_train – Training images

seg_test – Testing images

Each class is stored in a separate subfolder.

Methodology
1. Data Preprocessing

Images are loaded using OpenCV

Converted to grayscale

Resized to 64 × 64

Flattened into 1D feature vectors

2. Models Used

Decision Tree Classifier (baseline model)

Random Forest Classifier (ensemble model)

3. Training and Evaluation

Models are trained using the provided training set

Evaluated on the test set

Performance metrics include:

Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

4. Hyperparameter Experiment

Random Forest performance is evaluated by varying the number of trees (n_estimators)

Accuracy trends are visualized to study the effect of ensemble size

Results and Discussion

The Random Forest model consistently outperformed the Decision Tree classifier, demonstrating the advantage of ensemble learning over a single tree. Increasing the number of trees improved accuracy up to a certain point, after which the gains became marginal.

A key limitation of this approach is the use of flattened grayscale images instead of deep learning–based feature extraction, which could further improve performance on image data.

Project Structure
intel-image-classification/
│
├── data/
│   ├── seg_train/
│   └── seg_test/
│
├── Intel Image-Code.ipynb
├── README.md

Environment and Requirements

The project uses a single Python environment shared with other classification tasks.

Required libraries:

Python 3.10

NumPy

OpenCV

scikit-learn

Matplotlib

Seaborn

Jupyter Notebook

Install dependencies:

pip install numpy opencv-python scikit-learn matplotlib seaborn jupyter

How to Run

Clone the repository

Open Intel Image-Code.ipynb

Update dataset paths if required

Run the notebook from top to bottom

Author

Mahvish Ayub

