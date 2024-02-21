import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from zipfile import ZipFile
from io import BytesIO
import requests

# Download the ZIP file
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
response = requests.get(url)

# Extract the CSV file from the ZIP archive
with ZipFile(BytesIO(response.content)) as zip_file:
    csv_file = zip_file.open('bank-additional/bank-additional-full.csv')
    bank_data = pd.read_csv(csv_file, sep=';')

# Convert categorical variables to dummy variables
bank_data = pd.get_dummies(bank_data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'])

# Convert target variable to binary
bank_data['y'] = bank_data['y'].map({'yes': 1, 'no': 0})

# Split data into features and target variable
X = bank_data.drop('y', axis=1)
y = bank_data['y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Display classification report
print(classification_report(y_test, y_pred))

# Plot the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns)
plt.show()
