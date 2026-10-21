# 📊 Data Drift Early Warning System

## 📌 Project Overview

The Data Drift Early Warning System detects changes between
training data and new incoming data.

Data drift occurs when the distribution of new data becomes
different from the data used to train an ML model.

The system gives an early warning so that the ML model can
be checked or retrained.

---

## 🎯 Problem Statement

Machine Learning models are trained using historical data.
In real-world applications, incoming data can change over time.

If these changes are not detected, model performance may decrease.

This project detects data drift and displays the result
through a Streamlit dashboard.

---

## 💡 Solution

The system:

1. Uploads training data.
2. Uploads new incoming data.
3. Compares common features.
4. Calculates drift scores.
5. Shows Normal, Warning, or Drift status.
6. Provides recommendations.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Streamlit
- Docker
- Git
- GitHub
- Jenkins
- Kubernetes

---

## 📁 Project Structure

```text
data_drift_early_warning/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── training_data.csv
└── new_data.csv

DevOps CI/CD pipeline configured using Jenkins and Kubernetes.
Add-Content README.md "`nWebhook test"
Jenkins CI CD test

Final webhook test
