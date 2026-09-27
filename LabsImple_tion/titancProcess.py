import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = sns.load_dataset('titanic')[['survived','pclass','sex','age','sibsp','parch','fare','embarked']]

# Handle missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Encode categorical features
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['embarked'], drop_first=True)

# Features / target
X = df.drop('survived', axis=1)
y = df['survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
norm = MinMaxScaler()
X_train_norm = norm.fit_transform(X_train)
X_test_norm = norm.transform(X_test)

# Build and train ANN
ann = MLPClassifier(hidden_layer_sizes=(10,10), learning_rate_init=0.01,
                     momentum=0.9, max_iter=500, random_state=42)
ann.fit(X_train_norm, y_train)

# Query network
y_pred = ann.predict(X_test_norm)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Sample prediction:", ann.predict([X_test_norm[0]]), "Actual:", y_test.values[0])