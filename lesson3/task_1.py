import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

boston = load_boston()

data = boston.data

feature_names = boston.feature_names
target = boston.target

X = pd.DataFrame(data, columns=feature_names)
y = pd.DataFrame(target, columns=['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(X_train)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
check_test = pd.DataFrame({
    "y_test": y_test["price"],
    "y_pred": y_pred.flatten(),
})
check_test["error"] = check_test["y_pred"] - check_test["y_test"]
print(check_test.head())


print(r2_score(check_test.y_pred, check_test.y_test))
