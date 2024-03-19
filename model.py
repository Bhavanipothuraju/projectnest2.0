# model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

def train_model():
    # Load the dataset from Colab
    credit_card_data = pd.read_csv('creditcard_2023.csv')

    # Drop missing values
    credit_card_data.dropna(inplace=True)

    # Separate the data for analysis
    legit = credit_card_data[credit_card_data['Class'] == 0]
    fraud = credit_card_data[credit_card_data['Class'] == 1]

    # Under sampling
    legit_sample = legit.sample(n=492)
    new_dataset = pd.concat([legit_sample, fraud], axis=0)

    # Split the data into features (X) and target variable (y)
    X = new_dataset[['id', 'Amount']]  # Select only 'id' and 'Amount' columns
    y = new_dataset['Class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training - Logistic Regression
    model = LogisticRegression()

    # Impute missing values in X_train using mean imputation
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)

    # Fit the model

    model.fit(X_train, y_train)

    # Save feature names for later use
    feature_names = X.columns.tolist()

    return model, imputer, feature_names