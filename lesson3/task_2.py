from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

boston = load_boston()

data = boston.data

feature_names = boston.feature_names
target = boston.target

X = pd.DataFrame(data, columns=feature_names)
y = pd.DataFrame(target, columns=['price'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
model.fit(X_train, y_train.values[:, 0])
y_pred = model.predict(X_test)
check_test = pd.DataFrame({
    "y_test": y_test["price"],
    "y_pred": y_pred.flatten(),
})
check_test["error"] = check_test["y_pred"] - check_test["y_test"]
print(check_test.head())
print(r2_score(check_test.y_pred, check_test.y_test))
# RandomForestRegressor = 0.8479049999699443
#LinearRegresion = 0.6693702691495601
# Т.к значения близкие к нулю говорят о лучшей работе модели, RandomForestRegressor лучше