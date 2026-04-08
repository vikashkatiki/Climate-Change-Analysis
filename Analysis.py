import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

temp = pd.read_csv("temperature.csv")
co2 = pd.read_csv("co2.csv", comment='#', header=None)
co2.columns = ['year','month','day','decimal','CO2']

print(temp.columns)
print(co2.columns)

print(temp.shape)
print(co2.shape)
print(temp.head())
print(co2.head())

# Rename Mean → Temperature
temp.rename(columns={'Mean': 'Temperature'}, inplace=True)

# Parse the Year column (values like "1850-01")
temp['Date'] = pd.to_datetime(temp['Year'], errors='coerce')

# Extract numeric year
temp['Year'] = temp['Date'].dt.year

# Keep needed columns & clean
temp = temp[['Year','Temperature']].dropna()
temp['Year'] = temp['Year'].astype(int)

co2['Year'] = pd.to_numeric(co2['year'], errors='coerce')
co2 = co2[['Year','CO2']].dropna()
co2['Year'] = co2['Year'].astype(int)

temp_yearly = temp.groupby('Year', as_index=False)['Temperature'].mean()
co2_yearly  = co2.groupby('Year', as_index=False)['CO2'].mean()

print(temp_yearly.head())
print(co2_yearly.head())

start_year = max(temp_yearly['Year'].min(), co2_yearly['Year'].min())
end_year   = min(temp_yearly['Year'].max(), co2_yearly['Year'].max())

temp_yearly = temp_yearly[(temp_yearly['Year'] >= start_year) & (temp_yearly['Year'] <= end_year)]
co2_yearly  = co2_yearly[(co2_yearly['Year']  >= start_year) & (co2_yearly['Year']  <= end_year)]

data = pd.merge(temp_yearly, co2_yearly, on='Year', how='inner')

print(data.head())
print("Rows:", len(data))

print(data.info())
print(data.describe())

print("Mean:", data['Temperature'].mean())
print("Median:", data['Temperature'].median())
print("Variance:", data['Temperature'].var())

print("Correlation:\n", data.corr())


# ===============================
# 1. Temperature Trend
# ===============================
plt.figure(figsize=(10,5))
plt.plot(data['Year'], data['Temperature'], linewidth=2, label="Annual Mean")
plt.plot(data['Year'], data['Temperature'].rolling(5).mean(), linestyle='--', label="5-year Rolling")

plt.title("Global Temperature Trend", fontsize=14, weight='bold')
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()


# ===============================
# 2. CO2 Trend 
# ===============================
plt.figure(figsize=(10,5))
plt.plot(data['Year'], data['CO2'], linewidth=2)

plt.title("CO2 Concentration Over Time", fontsize=14, weight='bold')
plt.xlabel("Year")
plt.ylabel("CO2 (ppm)")
plt.tight_layout()
plt.show()

# ===============================
# 3. CO₂ vs Temperature (Regression)
# ===============================
plt.figure(figsize=(7,5))
sns.regplot(x='CO2', y='Temperature', data=data)

plt.title("CO2 vs Temperature", fontsize=14, weight='bold')
plt.xlabel("CO2 (ppm)")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()


# ===============================
# 4. Temperature Distribution
# ===============================
plt.figure(figsize=(7,5))
sns.histplot(data['Temperature'], kde=True)

plt.title("Temperature Distribution", fontsize=14, weight='bold')
plt.xlabel("Temperature (°C)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# ===============================
# 5. Temperature by Decade (BOXPLOT)
# ===============================
data['Decade'] = (data['Year']//10)*10

plt.figure(figsize=(8,5))
sns.boxplot(x='Decade', y='Temperature', data=data)

plt.title("Temperature by Decade", fontsize=14, weight='bold')
plt.xlabel("Decade")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ===============================
# 6. Correlation Heatmap
# ===============================
plt.figure(figsize=(6,5))
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', fmt=".2f")

plt.title("Correlation Heatmap", fontsize=14, weight='bold')
plt.tight_layout()
plt.show()

X = data[['Year','CO2']]
y = data['Temperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

future = pd.DataFrame([[2030, 420]], columns=['Year','CO2'])
prediction = model.predict(future)

print("Predicted Temperature in 2030:", prediction)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# ===============================
# 7. Actual vs Predicted (ML)
# ===============================
plt.figure(figsize=(7,5))
plt.scatter(y_test, y_pred)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

plt.title("Actual vs Predicted Temperature", fontsize=14, weight='bold')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()

# ===============================
# 8. Residual Plot
# ===============================
residuals = y_test - y_pred

plt.figure(figsize=(7,5))
plt.scatter(y_pred, residuals)

plt.axhline(0, linestyle='--')

plt.title("Residual Plot", fontsize=14, weight='bold')
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()


# ===============================
# 9. Pairplot 
# ===============================
sns.pairplot(data, diag_kind='kde')
plt.show()

Q1 = data['Temperature'].quantile(0.25)
Q3 = data['Temperature'].quantile(0.75)

IQR = Q3 - Q1

outliers = data[(data['Temperature'] < Q1 - 1.5*IQR) |
                (data['Temperature'] > Q3 + 1.5*IQR)]

print("Outliers:\n", outliers)

corr, p_value = stats.pearsonr(data['CO2'], data['Temperature'])

print("Correlation:", corr)
print("P-value:", p_value)

