# 🧠 Handwritten Digit Recognition with CNN using TensorFlow Keras

This project demonstrates how to use a **Convolutional Neural Network (CNN)** to recognize handwritten digits using the **MNIST dataset**.
It’s implemented in a Jupyter Notebook using **TensorFlow** and **Keras**, making it a beginner-friendly introduction to deep learning and computer vision.

---

## 📌 Project Overview

The goal is to train a CNN model to classify grayscale images of handwritten digits (0–9). The project involves loading the dataset, preprocessing the data, building and training the model, evaluating its accuracy, and visualizing predictions.

---

## 📌 Problem Statement

Manual digit recognition is prone to errors and not scalable. This project automates digit recognition using deep learning, offering a fast and reliable system for recognizing handwritten digits.

---


## 🧠 Solution Overview

Model: Convolutional Neural Network (CNN) built with TensorFlow/Keras

Purpose: Classifies handwritten digits (0 to 9) from grayscale images

Dataset: MNIST dataset (28x28 grayscale digit images)

Accuracy: Achieved over 98% test accuracy

---

## 📁 Dataset: MNIST


Name: MNIST Handwritten Digits
- **60,000** training images
- **10,000** testing images
- Each image is 28×28 pixels in grayscale
- Automatically loaded using:  
  python
  from tensorflow.keras.datasets import mnist

---
  
## 🛠️ Technologies Used
Python 🐍

Jupyter Notebook

TensorFlow / Keras

NumPy

Matplotlib

---

## ✨ Features

Automatic download of MNIST dataset

CNN architecture for image classification

Model training, evaluation, and prediction

Random digit prediction visualization

Confusion matrix and accuracy reporting

---

## 🧠 CNN Architecture Summary

Input: 28x28 grayscale image
↓
Conv2D (32 filters) + ReLU
↓
MaxPooling2D (2x2)
↓
Conv2D (64 filters) + ReLU
↓
MaxPooling2D (2x2)
↓
Flatten
↓
Dense (128 neurons) + ReLU
↓
Output (10 neurons, Softmax)

## 🚀 How to Run This Project
Clone the Repository

## bash
git clone https://github.com/SejalMandhan/Handwritten-Digit-Recognition-with-CNN-using-TensorFlow-Keras.git

## Install Dependencies

## bash

pip install -r requirements.txt

## Open the Notebook

## bash

jupyter notebook "Handwritten Digit Recognition with CNN using TensorFlow Keras.ipynb"

## Run All Cells

The notebook trains the model and shows test predictions.

## 📈 Example Output

Predicted: 7 | Actual: 7
Displays the digit image along with the model's prediction and ground truth label.

🙋‍♂️ Author Sejal Mandhan LinkedIn (www.linkedin.com/in/sejal-mandhan-a8b8302a6)
