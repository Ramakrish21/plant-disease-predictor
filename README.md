# 🥔 Plant Disease Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview
Agriculture plays a vital role in the global economy, but crop diseases significantly reduce yield and quality. This project is an **automated plant disease detection system** designed to assist farmers.

Using a **Convolutional Neural Network (CNN)**, the system identifies potato leaf diseases from images. It is deployed as a web application using **FastAPI**, allowing users to upload a leaf image and get an instant diagnosis along with a confidence score and severity assessment.

## 🚀 Features
* **Deep Learning Powered:** Custom CNN model trained on the Kaggle Potato Leaf Dataset.
* **5-Class Classification:**
    * Healthy
    * Early Blight (Mild)
    * Early Blight (Severe)
    * Late Blight (Mild)
    * Late Blight (Severe)
* **Real-time Prediction:** Instant results via FastAPI backend.
* **User-Friendly Interface:** Drag-and-drop image upload (HTML/CSS/JS).
* **Confidence Score:** Displays the probability of the predicted disease.

## 🛠️ Technology Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.x |
| **Deep Learning** | TensorFlow, Keras |
| **Backend API** | FastAPI, Uvicorn |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Image Processing** | NumPy, Pillow (PIL) |
| **Training** | Jupyter Notebook, Matplotlib |

## 🧠 Model Architecture
We implemented a **Sequential CNN** architecture optimized for image classification:
1.  **Preprocessing:** Resizing (256x256) and Rescaling (0-1).
2.  **Feature Extraction:** Multiple `Conv2D` layers followed by `MaxPooling2D` layers to capture patterns like leaf edges and spot textures.
3.  **Classification:** A `Flatten` layer feeds into `Dense` layers.
4.  **Output:** A Softmax layer with 5 neurons to predict the specific disease class.

**Performance:** The model achieved approximately **85% accuracy** on the test dataset.

## 📂 Project Structure

```bash
/Potato-Disease-Predictor
│
├── ai-model/
│   └── Model/
│       ├── potatoes_trained_model.h5   # Trained Model File
│       └── class_names.json            # List of classes
│
├── static/
│   ├── style.css                     # Frontend Styling
│   └── script.js                     # Frontend Logic
│
├── templates/
│   ├── index.html                    # Upload Page
│   └── result.html                   # Prediction Result Page
│
├── main.py                           # FastAPI Backend Application
├── requirements.txt                  # Python Dependencies
└── README.md                         # Project Documentation
```
📸 Screenshots
(You can upload screenshots of your web app here later)

🚀 How to Run Locally
1. Clone the Repository
   
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Run the Server

```bash
uvicorn main:app --reload
```

4. Access the App

```bash
Open your browser and go to: http://127.0.0.1:8000
```
