# Sentiment Analysis of IMDb Movie Reviews using DeepLearning(LSTM,Transformer,GRU)

## Project Overview

This repository contains a deep learning project focused on sentiment analysis of IMDb movie reviews. The goal is to predict whether a given movie review is positive or negative using classification models, including bi-directional LSTM, Transformer, and GRU architectures. The project evaluates and compares the performance of each model based on key metrics.

## Dataset

- **IMDb Movie Reviews**: The dataset consists of 50,000 movie reviews, split into 25,000 for training and 25,000 for testing. Each review is labeled as either positive or negative, making this a binary classification problem.

## Models Used

1. **Bi-directional LSTM**: This model used early stopping with a patience of 5 and achieved the best performance.
   - **Train Accuracy**: 0.96
   - **Test Accuracy**: 0.84

2. **Transformer Model**: Created from scratch using TensorFlow's Keras API (`tf.keras.Model`).
   - **Configuration**:
     - Embedding Dimension: 25
     - Number of Heads: 2
     - Sequence Length: 500
     - Vocab Size: 5000
     - Feed-forward Dimension: 64
     - Dense Layer: 64 units with Dropout of 0.3, followed by a Dense Layer with 1 unit.

3. **GRU Model**: 
   - **Evaluation Metrics**:
     - Precision: 0.8991
     - Recall: 0.8418
     - Accuracy: 0.8727

## Requirements

- Python 3.x
- TensorFlow
- scikit-learn
- Pandas
- Numpy

## Installation

To install the required libraries, run:

```bash
pip install -r requirements.txt
