import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Import Required Libraries and set style
sns.set_style('whitegrid')
np.random.seed(42)

# 2) Load the Dataset
data_url = 'https://raw.githubusercontent.com/huzaifsayed/Linear-Regression-Model-for-House-Price-Prediction/master/USA_Housing.csv'
df = pd.read_csv(data_url)

print('Dataset shape:', df.shape)
print(df.head())

# 3) Data Understanding and Pre-processing
print(df.info())
print(df.describe())

# Check missing values
print(df.isnull().sum())

# Drop non-numeric identifier column (Address) before modeling
df_model = df.drop('Address', axis=1)
print(df_model.head())

# 4) Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 7))
corr = df_model.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# 5) Define Features and Target
X = df_model.drop('Price', axis=1)
y = df_model['Price']

print('Feature matrix shape:', X.shape)
print('Target vector shape :', y.shape)
print(X.head())

# 6) Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print('X_train:', X_train.shape)
print('X_test :', X_test.shape)
print('y_train:', y_train.shape)
print('y_test :', y_test.shape)

# 7) Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('Model training completed.')

# 8) Model Coefficients and Intercept
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
})

print('Intercept:', lr_model.intercept_)
print(coeff_df)

# 9) Predictions and Evaluation
y_pred = lr_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE:  {mae:,.2f}')
print(f'MSE:  {mse:,.2f}')
print(f'RMSE: {rmse:,.2f}')
print(f'R2:   {r2:.4f}')

results_df = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})
print(results_df.head(10))

# 10) Plotting Predictions
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')

# Ideal line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.show()