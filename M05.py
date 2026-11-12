
# Project M05: Concrete Strength from Cement Amount

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# STEP 1: Load the data file
df = pd.read_csv("Concrete_Data_Yeh.csv")

print("Rows and columns:", df.shape)


# STEP 2: Clean the data
df = df.dropna(subset=["cement", "csMPa"])


# STEP 3: Choose input X and output y
X = df[["cement"]]
y = df["csMPa"]


# STEP 4: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# STEP 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# STEP 6: Test the model
y_pred = model.predict(X_test)

print("R2 score       :", round(r2_score(y_test, y_pred), 3))

print("Average error  :",
      round(mean_absolute_error(y_test, y_pred), 2), "MPa")

print("Slope m        :", round(model.coef_[0], 4))

print("Intercept c    :", round(model.intercept_, 2))


# STEP 7: Predict a new value
new = pd.DataFrame({"cement": [400]})

print(
    "Predicted strength with 400 kg cement:",
    round(model.predict(new)[0], 2),
    "MPa"
)


# STEP 8: Draw the graph
line_x = pd.DataFrame({
    "cement": sorted(X["cement"])
})

plt.scatter(
    X_test["cement"],
    y_test,
    s=12,
    alpha=0.6,
    label="Actual (test data)"
)

plt.plot(
    line_x["cement"],
    model.predict(line_x),
    color="red",
    label="Regression line"
)

plt.xlabel("Cement (kg per cubic metre)")
plt.ylabel("Compressive strength (MPa)")
plt.title("Concrete strength from cement amount")

plt.legend()
plt.show()
