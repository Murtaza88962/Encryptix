# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the train and test data
# Assuming the data is in TXT format and fields are separated by ':::'
train_df = pd.read_csv('train_data.txt', sep=':::', engine='python', header=None, names=['DESCRIPTION', 'GENRE'])
test_df = pd.read_csv('test_data.txt', sep=':::', engine='python', header=None, names=['DESCRIPTION'])

# Display the first few rows of the train data to understand its structure
print(train_df.head())

# Display the first few rows of the test data to understand its structure
print(test_df.head())

# Preprocess the data
# Remove any unnecessary whitespace
train_df['DESCRIPTION'] = train_df['DESCRIPTION'].str.strip()
test_df['DESCRIPTION'] = test_df['DESCRIPTION'].str.strip()

# Separate the features and target variable from the train data
X_train = train_df['DESCRIPTION']
y_train = train_df['GENRE']

# Use TF-IDF vectorizer to convert text data into numerical format
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)

# Display the shape of the transformed feature matrix
print(X_train_tfidf.shape)

# Train a Logistic Regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

# Prepare the test data
X_test = test_df['DESCRIPTION']
X_test_tfidf = tfidf.transform(X_test)

# Predict the genres for the test data
test_df['PREDICTED_GENRE'] = lr.predict(X_test_tfidf)

# Display the first few rows of the test data with predicted genres
print(test_df.head())

# Save the predictions to a new CSV file
test_df.to_csv('test_data_with_predictions.csv', index=False, sep=':::')

# Evaluate the model on the train data (for illustration purposes)
y_train_pred = lr.predict(X_train_tfidf)
print('Train Accuracy:', accuracy_score(y_train, y_train_pred))
print('Classification Report:', classification_report(y_train, y_train_pred))
