# 🎬 Sentiment Analysis Flask App

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0-green?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **Note:** This repository is a refactor of a University Project on Artificial Intelligence. It has been modernized to reflect Clean Code practices and organized as a Portfolio artifact.
## 📸 App Interface & Demo

<p align="center">
  <strong>Main Interface</strong><br>
  <img src="Form Review.png" width="80%" alt="Main Form Interface" style="border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);">
</p>

<p align="center">
  <strong>Prediction Results</strong><br>
  <img src="Positive Review.png" width="45%" alt="Positive Result">
  <img src="Negative Review.png" width="45%" alt="Negative Result">
</p>

## 📖 Project Overview

This application is a **Movie Review Classifier** that predicts whether a given review is **Positive** or **Negative**. 

Unlike standard ML models that require loading the entire dataset into RAM, this project implements **Out-of-Core Learning** using the `SGDClassifier` from Scikit-Learn. This allows the model to learn from massive datasets (like the IMDb dataset used here) in small batches, making it highly memory efficient.

### Key Capabilities
* **Real-Time Prediction:** Instant sentiment analysis via a Web Interface.
* **Active Learning Loop:** If the user flags a prediction as incorrect, the system captures the feedback and can re-train the model to improve future accuracy.
* **State Persistence:** Stores reviews and user feedback in a SQLite database.

## 🏗️ Architecture & Project Structure

The project follows a modular structure separating the logic, data, and presentation layers:

```text
sentiment-analysis-flask-app/
│
├── app.py              # 🚀 Main Application Controller (Flask)
│                       # Handles routing and serves the web interface.
│
├── vectorizer.py       # 🧠 Text Processing Logic
│                       # Contains the HashingVectorizer and tokenizer to convert 
│                       # raw text into numerical features for the model.
│
├── setup_repo.py       # 🛠️ Initialization Script
│                       # Generates the SQLite database and trains a lightweight 
│                       # dummy model for first-time setup and testing.
│
├── update.py           # 🔄 Model Maintenance
│                       # Script to re-train the model using new verified 
│                       # data collected in the SQLite database.
│
├── requirements.txt    # 📦 Dependencies
│                       # List of required Python libraries (Flask, Scikit-learn, etc.)
│
├── data/               # 💾 Data Layer
│   ├── reviews.sqlite  # Database storing user reviews and feedback.
│   └── pkl_objects/    # Serialized objects (Classifier & Stopwords).
│
├── static/             # 🎨 Static Assets
│   └── style.css       # Custom CSS for the web interface.
│
└── templates/            # 🖥️ UI Templates
    ├── _formhelpers.html # 🧩 Jinja2 Macro. Reusable component to render form fields and handle validation errors.
    ├── reviewform.html   # Main input form.
    ├── results.html      # Prediction result display.
    └── thanks.html       # Feedback confirmation page.
    
```

## 🚀 How to Run Locally
Follow these steps to get the application running on your machine:

**1. Clone the repository** 
```bash
git clone [https://github.com/JorgeGimenezGarcia/sentiment-analysis-flask-app.git](https://github.com/JorgeGimenezGarcia/sentiment-analysis-flask-app.git)
cd sentiment-analysis-flask-app
```
**2. Create and Activate Virtual Environment** 
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```
**3. Install Dependencies** 
```bash
pip install -r requirements.txt
```
**4. Initialize the Project (Important) Run the setup script to generate the database and a lightweight demo model**
```bash
python app.py
```
> Note: This script trains a dummy model for demonstration purposes ensuring the app runs immediately without downloading large datasets. For production accuracy, the full IMDb dataset should be used.

**5. Run the application** 
```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`.

## 🧠 Technical Deep Dive
The Model: Stochastic Gradient Descent (SGD)
The core of this app is an SGDClassifier. We use the partial_fit method, which allows the model to update its weights incrementally.

  * Vectorizer: HashingVectorizer (2^21 features) is used to convert text to vectors independently of the dataset size, eliminating the need to store a massive vocabulary dictionary in memory.

  * Update Mechanism: The update.py script serves as a maintenance tool to fetch verified feedback from the SQLite DB and refine the model over time.

Author: Jorge Giménez García
Senior Data Engineer
