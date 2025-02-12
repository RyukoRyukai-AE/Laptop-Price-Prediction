import joblib
import pandas as pd

new_data = pd.read_csv('new_data.csv')

y = new_data['Price ($)']
X = new_data.drop(columns=['Price ($)'])

transformer = joblib.load('Machine_Learning/Supervised_Learning/Regression/Laptop_Price_Prediction/model/Transformer.pkl')
X_transformed = transformer.fit_transform(X)

model = joblib.load('Machine_Learning/Supervised_Learning/Regression/Laptop_Price_Prediction/model/laptop_price_prediction.pkl')
y_pred = model.predict(X_transformed)

print(y_pred)