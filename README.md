# Automated Essay Scoring

This repository contains my code for the Kaggle competition on automated essay scoring.

## Introduction
Automated essay scoring (AES) is the process of evaluating and scoring essays using computer algorithms. In this competition, the goal is to develop a model that can accurately predict the scores of essays based on their content.

## Features
- Preprocessing: The text data is preprocessed using tokenization, removal of stopwords, punctuation, and lowercasing.
- Feature Engineering: The TfidfVectorizer is used to convert text data into numerical features.
- Model: Random Forest Classifier is used as the predictive model.
- Grid Search: Grid search is performed to optimize hyperparameters of the Random Forest Classifier.
- Evaluation Metric: Cohen's Kappa Score with quadratic weights is used as the evaluation metric.

## How it Works
1. **Data Loading**: Load the training and test datasets.
2. **Preprocessing**: Preprocess the text data by tokenization, removal of stopwords, punctuation, and lowercasing.
3. **Feature Engineering**: Convert the preprocessed text data into numerical features using TfidfVectorizer.
4. **Model Training**: Train a Random Forest Classifier using the training data.
5. **Hyperparameter Tuning**: Perform grid search to optimize hyperparameters of the Random Forest Classifier.
6. **Model Evaluation**: Evaluate the model performance using Cohen's Kappa Score.
7. **Prediction**: Make predictions on the test dataset using the trained model.
8. **Submission**: Generate a submission file for the Kaggle competition.

## Requirements
- Python 3
- pandas
- scikit-learn
- nltk

## Usage
1. Clone the repository: `git clone https://github.com/PeterKayode/Automated-Essay-Scoring`
2. Navigate to the project directory: `cd Automated-Essay-Scoring`
3. Install the required packages: `pip install -r requirements.txt`
4. Run the code: `Automated Essay Scoring Kaggle Competition.ipynb`

## Dataset
The dataset for this competition can be found on Kaggle: [Automated Essay Scoring Dataset](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/data)
