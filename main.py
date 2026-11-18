import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, root_mean_squared_error
model = RandomForestRegressor(
    n_jobs=-1,
    n_estimators=3000,
    max_depth=16,
    min_samples_leaf=1,
    min_samples_split=4,
    random_state=42)
data = pd.read_csv("data.csv",index_col=False)
x,y = data, data['Price']
x = x.drop(columns=['Airline','Source','Destination','Price'])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
pred = model.predict(x_test)
r2 = r2_score(y_test,pred)
mape = mean_absolute_percentage_error(y_test,pred)
mae = mean_absolute_error(y_test,pred)
mse = mean_squared_error(y_test,pred)
rmse = root_mean_squared_error(y_test,pred)
print(f'R2: {r2}')
print(f'MAPE: {mape}')
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
joblib.dump(model,'predictor.pkl',compress=3)