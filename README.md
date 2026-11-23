# 🫁 Chest X-ray Pneumonia Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-3.15-red?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Status-Development-yellow" alt="Project Status">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  An AI-powered deep learning application that analyzes chest X-ray images and predicts whether they indicate NORMAL or PNEUMONIA.
</p>

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [Project Objectives](#-project-objectives)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [System Architecture](#-system-architecture)
- [Installation and Setup](#-installation-and-setup)
- [Model Training](#-model-training)
- [Model Evaluation](#-model-evaluation)
- [Grad-CAM Visualization](#-grad-cam-visualization)
- [Run the Streamlit Application](#-run-the-streamlit-application)
- [Docker Deployment](#-docker-deployment)
- [Git LFS Setup](#-git-lfs-setup)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## 🧠 About the Project

The **Chest X-ray Pneumonia Detection System** is a deep learning-based computer vision project developed to classify chest X-ray images into two categories:

- **NORMAL** – Chest X-ray images without pneumonia.
- **PNEUMONIA** – Chest X-ray images indicating pneumonia.

The project uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to learn visual patterns from chest X-ray images.

A Streamlit-based web application provides an interactive interface where users can upload an X-ray image and receive a model-generated prediction.

The project also explores model evaluation, data preprocessing, and Grad-CAM visualization to help understand the regions influencing the model's predictions.

---

## 🎯 Project Objectives

- Develop a CNN-based image classification model for pneumonia detection.
- Preprocess and organize chest X-ray image datasets.
- Train and evaluate a deep learning model.
- Classify images into NORMAL and PNEUMONIA categories.
- Visualize model predictions using Grad-CAM.
- Build an interactive Streamlit dashboard.
- Containerize the application using Docker.
- Maintain a reproducible and organized machine learning workflow.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🩻 Chest X-ray Upload | Upload chest X-ray images through the web application. |
| 🤖 CNN Prediction | Predict NORMAL or PNEUMONIA using a trained CNN model. |
| 📊 Prediction Confidence | Display the model's prediction score. |
| 📈 Dataset Insights | Explore dataset distribution and image information when the dataset is available. |
| 🔥 Grad-CAM | Visualize image regions influencing model predictions. |
| 🖥️ Interactive Dashboard | User-friendly Streamlit interface. |
| 🐳 Docker Support | Run the application inside a Docker container. |
| 🧪 Model Evaluation | Evaluate classification performance using standard metrics. |

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development |
| NumPy | Numerical operations |
| Pandas | Data manipulation |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation metrics |
| Pillow | Image processing |
| Streamlit | Interactive web application |
| Docker | Application containerization |
| Git & GitHub | Version control and project hosting |
| Git LFS | Large model file storage |
| Pytest | Testing |

---

## 📂 Dataset

This project uses a chest X-ray image dataset organized into three subsets:

- Training dataset
- Validation dataset
- Testing dataset

Each subset contains two classes:

- NORMAL
- PNEUMONIA

### Dataset Structure

```text
chest_xray/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Dataset Preprocessing

The preprocessing workflow includes:

1. Loading chest X-ray images.
2. Resizing images to 224 × 224 pixels.
3. Converting images into RGB format.
4. Normalizing pixel values.
5. Assigning class labels.
6. Preparing NumPy arrays for model training and evaluation.

**Class Labels:**

| Class | Label |
|---|---:|
| NORMAL | 0 |
| PNEUMONIA | 1 |

The original dataset is not included in this repository because of its size.

---

## 📁 Project Structure

```text
Chest-Xray-Pneumonia-Detection-System/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── raw/
│       └── chest_xray/
│
├── models/
│   ├── best_custom_cnn.keras
│   └── final_custom_cnn.keras
│
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 02_data_preprocessing.py
│   ├── 03_CNN_model_training.py
│   ├── 04_model_evaluation.py
│   ├── 05_gradcam_visualization.py
│   └── 06_prediction.py
│
├── outputs/
│
├── processed_data/
│
├── screenshots/
│
├── src/
│
├── tests/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dockerignore
├── .gitattributes
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

*Note: Some folders contain generated data or local files and may not be included in the GitHub repository.*

---

## 🏗️ System Architecture

The application follows this general machine learning workflow:

```text
          Chest X-ray Dataset
                  |
                  v
          Data Exploration
                  |
                  v
          Image Preprocessing
                  |
                  v
          CNN Model Training
                  |
                  v
          Model Evaluation
                  |
                  v
          Trained CNN Model
                  |
                  v
          Streamlit Web Application
                  |
                  v
          Upload Chest X-ray Image
                  |
                  v
          Image Preprocessing
                  |
                  v
          CNN Model Prediction
                  |
                  v
          NORMAL / PNEUMONIA
                  |
                  v
          Prediction Visualization
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/User_Name/chest-xray-pneumonia-detection-system.git
```

### 2. Navigate to the Project Directory

```bash
cd chest-xray-pneumonia-detection-system
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Download the Dataset

Place the dataset in:

```text
data/raw/chest_xray/
```

Ensure the dataset follows the folder structure described above.

### 7. Obtain the Trained Model

If the model is stored using Git LFS, install Git LFS and retrieve the model:

```bash
git lfs install
git lfs pull
```

Ensure the model file is available at the path expected by the application.

---

## 🧠 Model Training

The project uses a Convolutional Neural Network (CNN) for binary image classification.

The training workflow includes:

1. Loading preprocessed training data.
2. Preparing training and validation datasets.
3. Defining the CNN architecture.
4. Compiling the model.
5. Training the model.
6. Monitoring validation performance.
7. Saving the trained model.

Run the training script:

```bash
python notebooks/03_CNN_model_training.py
```

The trained model is saved in the `models/` directory.

---

## 📊 Model Evaluation

The model evaluation workflow assesses classification performance on the test dataset.

The evaluation script can be executed using:

```bash
python notebooks/04_model_evaluation.py
```

Evaluation metrics may include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics help analyze the model's ability to distinguish between NORMAL and PNEUMONIA images.

*Actual performance values should be added after running the final evaluation.*

---

## 🔥 Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize image regions that contribute to a CNN prediction.

This visualization can help interpret which parts of a chest X-ray influence the model's output.

Run:

```bash
python notebooks/05_gradcam_visualization.py
```

Grad-CAM is an interpretability aid and does not establish that the model is clinically correct.

---

## 🔮 Prediction

The prediction script can be used to test an individual chest X-ray image.

Run:

```bash
python notebooks/06_prediction.py
```

Provide the appropriate image path according to the script's input requirements.

The model predicts one of the following classes:

- NORMAL
- PNEUMONIA

---

## 🖥️ Run the Streamlit Application

Start the interactive dashboard:

```bash
streamlit run app/streamlit_app.py
```

Open your browser:

```text
http://localhost:8501
```

### Application Workflow

1. Launch the Streamlit application.
2. Open the dashboard.
3. Upload a chest X-ray image.
4. Allow the model to process the image.
5. View the predicted classification.
6. Review the prediction score and available visualizations.

---

## 🐳 Docker Deployment

The application can be containerized using Docker.

### 1. Build the Docker Image

Run from the project root:

```bash
docker build -t chest-xray-pneumonia .
```

### 2. Run the Docker Container

```bash
docker run -d -p 8501:8501 --name pneumonia-container chest-xray-pneumonia
```

### 3. Open the Application

```text
http://localhost:8501
```

### 4. View Container Logs

```bash
docker logs pneumonia-container
```

### 5. Stop the Container

```bash
docker stop pneumonia-container
```

### 6. Remove the Container

```bash
docker rm pneumonia-container
```

---

## 📦 Git LFS Setup

Trained `.keras` model files can be larger than GitHub's standard 100 MB file limit.

Git Large File Storage (Git LFS) is used to manage large model files.

### Install Git LFS

Download:

https://git-lfs.com/

### Configure Git LFS

```bash
git lfs install
git lfs track "*.keras"
```

Add and commit the tracking configuration:

```bash
git add .gitattributes
git add models/
git commit -m "Add trained CNN model using Git LFS"
```

Push changes:

```bash
git push origin main
```

For an existing Git history containing large files, migrate the history to LFS before pushing.

---

## 🧪 Testing

Run the project's tests using:

```bash
python -m pytest -v
```

This executes the available automated tests in the `tests/` directory.

---

## 🚀 Future Improvements

- Improve model generalization using additional training data.
- Experiment with transfer learning architectures.
- Optimize prediction speed and memory usage.
- Improve Grad-CAM visualizations.
- Add more detailed evaluation reports.
- Improve dashboard design and user experience.
- Add model monitoring and experiment tracking.
- Explore deployment on cloud platforms.
- Improve automated testing and continuous integration.

---

## ⚠️ Disclaimer

This project is developed for educational and research purposes only.

The system is an experimental AI-based image classification application. It is not a medical diagnostic device and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

Predictions may be incorrect. Always consult qualified healthcare professionals for medical evaluation.

---

## 👨‍💻 Author

**Prathamesh Kasande**

GitHub: [@PrathameshKasande](https://github.com/PrathameshKasande)

Project Repository: [Chest X-ray Pneumonia Detection System](https://github.com/PrathameshKasande/chest-xray-pneumonia-detection-system)

---

## ⭐ Support

If you find this project interesting or useful, consider giving the repository a ⭐ on GitHub.

Thank you for visiting!

<p align="center">
  <b>Built with Python, TensorFlow, Keras and Streamlit ❤️</b>
</p>
