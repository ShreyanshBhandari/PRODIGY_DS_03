Bank Marketing Prediction with Decision Trees

This repository contains Python code for predicting whether a customer will purchase a product or service based on their demographic and behavioral data using a decision tree classifier. The code utilizes the Bank Marketing dataset from the UCI Machine Learning Repository.

Dataset:
The dataset used in this project is the Bank Marketing dataset, which contains information about marketing campaigns of a Portuguese banking institution. It includes various features such as age, job, marital status, education, etc., and the target variable is whether the customer subscribed to a term deposit or not.

Technologies Used:

Python
scikit-learn
pandas
matplotlib

Steps:

Data loading and preprocessing: The dataset is loaded and preprocessed by converting categorical variables to dummy variables and encoding the target variable.
Model training: A decision tree classifier is trained on the preprocessed data.
Model evaluation: The trained classifier is evaluated using accuracy, precision, recall, and F1-score.
Visualization: The decision tree is visualized to understand the decision boundaries learned by the classifier.
Usage:

Clone the repository.
Install the required libraries (pip install -r requirements.txt).
Run the Python script (predict_bank_marketing.py).
View the output including model performance metrics and the decision tree plot
