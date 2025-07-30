# Import necessary libraries
import pandas as pd
import seaborn as sns
import joblib as dump
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('data/iris.csv')

# Preview dataset
print(data.shape)
print(data.head())

# Unnecessary Id column is there
data.drop('Id', axis=1, inplace=True)

# Updated DataFrame
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Encode target labels into numeric format
le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])

# Updated DataFrame
print(data.head())
print(data.sample(5))

# Pair-plot to visualize relationships between all feature pairs
sns.pairplot(data=data, hue='Species')
plt.show()

# Split the dataset into features (x) and target (y)
x = data.drop('Species', axis=1)
y = data['Species']

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Display few samples
print(x_train.head())
print(x_test.head())
print(y_train.head())
print(y_test.head())

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=100)
model.fit(x_train, y_train)

# Predict the target values for the test set
y_pred = model.predict(x_test)
print(y_pred)

# Evaluate and display the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

# Print the detailed classification report
report = classification_report(y_test, y_pred)
print(report)

dump.dump(model, 'iris_model.pkl')
