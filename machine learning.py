
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# convert xlsx data to csv
read_file=pd.read_excel(r'C:\Users\reshm\Desktop\Chinnu\zoho\Rotten_Tomatoes_Movies3.xlsx')
read_file.to_csv(r'C:\Users\reshm\Desktop\Chinnu\zoho\ha.csv',index=None,header=True)

# Load the dataset
data = pd.read_csv(r'C:\Users\reshm\Desktop\Chinnu\zoho\ha.csv')
data.info()  
data.isnull().sum() 
data = data.dropna()

# Define feature matrix X and target vector y
X = data.drop('audience_rating', axis=1)
y = data['audience_rating']
print(X)
print(y)

# Identify numerical and categorical columns
num_features = ["runtime_in_minutes", "tomatometer_rating", "tomatometer_count"]
cat_features = ["genre", "rating"]
print(data.columns)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.head())
print(y_train.head())

# Define preprocessing pipeline
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features),
    ]
)

# Build the complete pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the evaluation metrics
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

