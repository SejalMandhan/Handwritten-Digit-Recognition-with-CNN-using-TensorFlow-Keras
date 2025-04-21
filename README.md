# 🧠 Handwritten Digit Recognition with CNN using TensorFlow Keras

This project demonstrates how to use a **Convolutional Neural Network (CNN)** to recognize handwritten digits using the **MNIST dataset**.
It’s implemented in a Jupyter Notebook using **TensorFlow** and **Keras**, making it a beginner-friendly introduction to deep learning and computer vision.

## 📌 Project Overview

The goal is to train a CNN model to classify grayscale images of handwritten digits (0–9). The project involves loading the dataset, preprocessing the data, building and training the model, evaluating its accuracy, and visualizing predictions.

---

## 📁 Dataset: MNIST

- **60,000** training images
- **10,000** testing images
- Each image is 28×28 pixels in grayscale
- Automatically loaded using:  
  python
  from tensorflow.keras.datasets import mnist
  
## 🛠️ Technologies Used
Python 🐍

Jupyter Notebook

TensorFlow / Keras

NumPy

Matplotlib

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

## 🎯 Features
Trains a CNN from scratch using Keras

Predicts digits from random MNIST test samples

Visual output with matplotlib

Includes actual vs. predicted digit display

## 📈 Example Output

Predicted: 7 | Actual: 7
Displays the digit image along with the model's prediction and ground truth label.

🙋‍♂️ Author Sejal Mandhan LinkedIn (www.linkedin.com/in/sejal-mandhan-a8b8302a6)
