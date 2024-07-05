import numpy as np
import pandas as pd
import sklearn.preprocessing
from geopy.distance import geodesic
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.linear_model import Lasso
import re

data = pd.read_csv("ApartmentRentPrediction.csv")
# start preprocessing
# handle nulls
missing_values = data.isnull().sum()
# print("Missing values per column:")
# print(missing_values)

# fill numerical values with median
numerical_col = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude']
for col in numerical_col:
    median_value = data[col].median()
    data[col].fillna(median_value, inplace=True)

# fill  categorical values with mode
categorical_col = ['category', 'amenities', 'pets_allowed', 'address', 'cityname', 'state']
for col in categorical_col:
    mode_value = data[col].mode()[0]
    data[col].fillna(mode_value, inplace=True)


# print(data.isnull().sum())
# $275 Monthly|Weekly
def extract_numerical_value(value):
    numerical_part = re.findall(r'\d[\d,.]*',value)
    if numerical_part:
        numerical_value = ''.join(numerical_part)
        return int(numerical_value.replace(',', ''))
    else:
        return None


data['price_display'] = data['price_display'].apply(extract_numerical_value)

# data['price_display'] = data['price_display'].str.replace('$', '').str.replace(',', '').astype(float)


# Encoding => convert categorical data to numerical
categorical_features = ['body', 'title', 'category', 'amenities', 'currency', 'fee', 'has_photo', 'pets_allowed',
                        'price_type', 'address', 'cityname', 'state', 'source']
label_encoder = preprocessing.LabelEncoder()
for col in categorical_features:
    data[col] = label_encoder.fit_transform(data[col])

# feature_scaling => standardization mean=0, sd=1
# notes : id should be normalized ? gonna drop it later anyway
numerical_feature = ['id', 'bathrooms', 'bedrooms', 'price', 'price_display', 'square_feet', 'latitude', 'longitude']
scaler = preprocessing.StandardScaler()
data[numerical_col] = scaler.fit_transform(data[numerical_col])
#print(data['has_photo'][6])
# print(data['time'].dtype)

# Outlier detection and handling
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
data = data[~outliers.any(axis=1)]

# feature selection using lasso
X = data.drop(columns=['id', 'time','body','title','price', 'price_display'])
y = data['price_display']

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
selected_features = X.columns[lasso_reg.coef_ != 0]
X_selected = X[selected_features]
print(selected_features)

#selector = SelectKBest(score_func=mutual_info_regression, k=10)
#X_selected = selector.fit_transform(X, y)
#selected_indices = selector.get_support(indices=True)

# Get selected feature names
#selected_feature_names = X.columns[selected_indices]

# Print selected feature names
#print("Selected Features:")
#print(selected_feature_names)

# create the model

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Linear regression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred_linear = linear_reg.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Polynomial regression
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("Mean Squared Error (Linear Regression):", mse_linear)
print("Mean Squared Error (Polynomial Regression):", mse_poly)

print("accuracy linear : " , r2_linear)
print("accuracy poly : " , r2_poly)



