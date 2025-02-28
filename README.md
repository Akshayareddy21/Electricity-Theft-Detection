# Electricity Theft Detection Using Deep Learning

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Files](#project-files)
  - [Code and Models](#code-and-models)
  - [Data Files](#data-files)
  - [Documentation](#documentation)
  - [Execution Script](#execution-script)
- [Installation & Setup](#installation--setup)
  - [Requirements](#requirements)
  - [Running the Project](#running-the-project)
- [Features](#features)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
This project focuses on detecting electricity theft cyber-attacks in renewable distributed generation systems using deep learning models such as:
- **Deep Feed Forward Neural Network (DNN)**
- **Recurrent Neural Network with Gated Recurrent Unit (RNN-GRU)**
- **Convolutional Neural Network (CNN)**

These models analyze smart meter data to identify potential fraudulent activities. CNN provides the highest accuracy in detecting electricity theft.

## Dataset
The dataset used in this project comes from Kaggle:
[Electricity Theft Detection Dataset](https://www.kaggle.com/mrmorj/fraud-detection-in-electricity-and-gas-consumption?select=client_train.csv)

It contains electricity consumption and generation details, with class labels:
- **0**: No Attack (Normal)
- **1**: Attack (Electricity Theft)

## Project Files
### Code and Models
- **`ElectricityTheftDetection.py`** - Main script for data processing, model training, and prediction.
- **`cnn_model.json`** & **`cnn_model_weights.h5`** - Pre-trained CNN model.
- **`gru_model.json`** & **`gru_model_weights.h5`** - Pre-trained GRU model.
- **`model.json`** & **`model_weights.h5`** - Pre-trained DNN model.

### Data Files
- **`ElectricityTheft.csv`** - Sample dataset for training and testing.
- **`test.csv`** - Sample test dataset.
- **`datasetlink.txt`** - Contains a link to the dataset source.

### Documentation
- **`SCREENS.docx`** - Contains screenshots of the application in action.
- **`A14. Electricity Theft docx.docx`** - Detailed project documentation, including background, methodology, and results.

### Execution Script
- **`run.bat`** - Batch file to run the application.

## Installation & Setup
### Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib
```

### Running the Project
1. Clone the repository:
```bash
git clone https://github.com/your-username/electricity-theft-detection.git
cd electricity-theft-detection
```
2. Run the Python script:
```bash
python ElectricityTheftDetection.py
```
3. If using the GUI, double-click `run.bat` to start the application.

## Features
- **Upload Dataset**: Load electricity theft dataset.
- **Preprocess Dataset**: Cleans data, handles missing values, and converts categorical data to numerical format.
- **Train Models**:
  - DNN for theft detection.
  - RNN-GRU for sequence-based anomaly detection.
  - CNN for feature extraction and classification.
- **Predict Electricity Theft**: Uses the trained models to detect fraudulent activities.
- **Generate Comparison Graphs**: Visualizes model performance.

## Results
- **CNN achieved 95.98% accuracy.**
- **DNN achieved 94.24% accuracy.**
- **GRU achieved 40.02% accuracy.**

## Future Improvements
- Incorporating additional features to enhance detection accuracy.
- Implementing real-time theft detection using IoT-based smart meters.
- Expanding dataset diversity for better generalization.

## Acknowledgments
- The dataset was sourced from Kaggle.
- Inspired by research on electricity theft detection using deep learning.

