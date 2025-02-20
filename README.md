
# Online Payment Fraud Detection

## Overview
This project focuses on detecting fraudulent transactions in an online payment dataset using a **Decision Tree Classifier**. The goal is to analyze transaction data, preprocess it, train a machine learning model, and evaluate its performance in identifying fraudulent activities.


## Dataset
The dataset used in this project contains information about online transactions, including:
- **Transaction Type**: CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT.
- **Amount**: The amount of the transaction.
- **isFraud**: A binary column indicating whether the transaction is fraudulent (1) or not (0).

You can download the dataset from [GitHub Releases](PASTE_LINK_HERE).


## Steps
1. **Load and Explore the Dataset**:
   - Load the dataset using `pandas`.
   - Check the shape of the dataset, missing values, and the distribution of transaction types.

2. **Data Visualization**:
   - Use `Plotly` to create a pie chart showing the distribution of transaction types.

3. **Data Preprocessing**:
   - Map categorical transaction types to numerical values.
   - Convert the `isFraud` column into a binary classification label.

4. **Model Training**:
   - Split the data into training and testing sets.
   - Train a **Decision Tree Classifier** using `scikit-learn`.

5. **Model Evaluation**:
   - Evaluate the model using **accuracy**, **classification report**, and **confusion matrix**.

6. **Make Predictions**:
   - Use the trained model to predict whether a new transaction is fraudulent.


## Code Structure
The project consists of the following files:
- `fraud_detection.py`: The main Python script containing the code for data analysis, preprocessing, model training, and evaluation.
- `README.md`: This file, providing an overview of the project.
- `requirements.txt`: A list of Python libraries required to run the project.
- `data/onlinefraud.csv.zip`: The dataset used in the project.


## Requirements
To run this project, you need the following Python libraries:
- pandas
- numpy
- plotly
- scikit-learn

Install the dependencies using:
```bash
pip install -r requirements.txt
# Online-Payment-Fraud-Detection
 2ceed12c88175539213f467694a05936765ec07c
