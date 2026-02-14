import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


#  Load Dataset

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseValue"] = housing.target

print("Dataset Shape:", df.shape)


# Feature Engineering

df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]
df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]
df["PopulationPerHousehold"] = df["Population"] / df["AveOccup"]

#  Separate Features & Target

X = df.drop("MedHouseValue", axis=1)
y = df["MedHouseValue"]

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Ridge Regression Model

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)


#  Make Predictions

y_pred = model.predict(X_test)


#Evaluate Model

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("R2 Score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)

import pickle

# Save model
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved successfully!")
