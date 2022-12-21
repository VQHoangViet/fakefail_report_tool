# building a model to predict fake fail index
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import pandas as pd

x = pd.read_csv(r"G:\.shortcut-targets-by-id\1AjwfMQyyeImg87H7oludOhN6s_HIj6RJ\VN-QA\29. QA - Data Analyst\FakeFail\final_data_monthly\final_driver_data 2022_12.csv")
# split data
x_train, x_test, y_train, y_test = train_test_split(x[['total_attempt', 'total_fake_fail_attempt', 'total_fake_fail_orders', 'total_orders', 'Total orders reach LM hub']].fillna(0), x['original_FF_index'].fillna(0), test_size=0.2, random_state=42)

# build model
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(x_train, y_train)

# predict
y_pred = model.predict(x_test)

# The coefficients
print('Coefficients: \n', model.steps[1][1].coef_)
# The mean squared error
print('Mean squared error: %.2f'
    % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
    % r2_score(y_test, y_pred))

#acuracy
print('Accuracy: %.2f' % model.score(x_test, y_test))
# plot
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()

  