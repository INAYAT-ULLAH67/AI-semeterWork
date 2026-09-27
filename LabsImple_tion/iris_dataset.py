import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("iris_dataset.csv")
X = df.drop(columns=["target", "species"])
Y = df["target"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

norm = MinMaxScaler().fit(X_train)

print("Min values:", norm.data_min_)
print("Max values:", norm.data_max_)

X_train_norm = norm.transform(X_train)
print("First 5 scaled rows:")
print(X_train_norm[:5])