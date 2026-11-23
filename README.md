# Chest X-ray Pneumonia Detection

A deep learning project for binary classification of chest X-ray images
into NORMAL and PNEUMONIA categories.

## Project Overview

This project explores computer vision techniques for chest X-ray
image classification using a custom Convolutional Neural Network (CNN).

It includes:
- Dataset exploration and preprocessing
- CNN model training
- Model evaluation
- Image prediction
- Grad-CAM visualization
- Streamlit interface

## Technology Stack

- Python
- TensorFlow / Keras
- NumPy and Pandas
- OpenCV
- Matplotlib
- Scikit-learn
- Streamlit
- Docker

## Project Structure

```text
app/          Streamlit application
src/          Data, model, training and prediction code
data/         Local dataset
models/       Trained model files
notebooks/    Exploration and experiments
outputs/      Evaluation results and visualizations
tests/        Automated tests

```text

## Dataset
Dataset: Chest X-Ray Images (Pneumonia)

Source:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Download the dataset from Kaggle and place the extracted
chest_xray folder inside data/raw/.

The dataset is not included in this repository.
Installation
py -3.11 -m venv venv

.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt


Train the CNN
python src/train.py

Evaluate the CNN
python src/evaluate.py

Run the application
python -m streamlit run app/streamlit_app.py

Run tests
python -m pytest -v

Limitations and Disclaimer
This project is intended for educational and research purposes only.
It is not a medical device and must not be used for clinical diagnosis
or treatment decisions. Model outputs are experimental and may be
incorrect or biased.
