"""
Time-series stock predictor with CLI argument parsing.

Reads a CSV file with columns: Date, Close
Creates lag features to predict future closing prices.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# -----------------------
# Command-line arguments
# -----------------------
parser = argparse.ArgumentParser(
    description="Time-series stock price predictor"
)

parser.add_argument(
    "--csv",
    type=str,
    default="AAPL.csv",
    help="Path to the input CSV file (default: AAPL.csv)"
)

parser.add_argument(
    "--days",
    type=int,
    default=5,
    help="Number of future days to predict (default: 5)"
)

args = parser.parse_args()


# -----------------------
# Config
# -----------------------
CSV_PATH = args.csv
DATE_COL = "Date"
TARGET_COL = "Close"
LAGS = 5
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
PREDICT_N_DAYS = args.days


# -----------------------
# Validate arguments
# -----------------------
if PREDICT_N_DAYS <= 0:
    raise ValueError("The number of prediction days must be greater than 0.")


# -----------------------
# Load & prepare data
# -----------------------
df = pd.read_csv(CSV_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL])

df = df.sort_values(DATE_COL).reset_index(drop=True)

# Ensure target exists and drop rows missing the target
df = df[[DATE_COL, TARGET_COL]].dropna().reset_index(drop=True)


# -----------------------
# Create lag features
# -----------------------
# Create lag features: Close_t-1 ... Close_t-LAGS
for lag in range(1, LAGS + 1):
    df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

# Remove rows with missing lag values
df = df.dropna().reset_index(drop=True)


# -----------------------
# Features and target
# -----------------------
feature_cols = [
    f"lag_{lag}" for lag in range(1, LAGS + 1)
]

X = df[feature_cols].copy()
y = df[TARGET_COL].copy()
dates = df[DATE_COL].copy()


# -----------------------
# Train/test split
# -----------------------
split_idx = int(len(df) * TRAIN_RATIO)

X_train = X.iloc[:split_idx].values
X_test = X.iloc[split_idx:].values

y_train = y.iloc[:split_idx].values
y_test = y.iloc[split_idx:].values

dates_train = dates.iloc[:split_idx]
dates_test = dates.iloc[split_idx:]


# -----------------------
# Scale features
# -----------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------
# Fit model
# -----------------------
model = LinearRegression()

model.fit(X_train_scaled, y_train)


# -----------------------
# Evaluate model
# -----------------------
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")


# -----------------------
# Forecast next N days
# -----------------------
# Start from the most recent lag features
last_known = df.iloc[-1][feature_cols].values.astype(float)

future_preds = []

current_lags = last_known.copy()

for i in range(PREDICT_N_DAYS):

    # Scale the current lag values
    scaled = scaler.transform(
        current_lags.reshape(1, -1)
    )

    # Predict the next closing price
    pred = model.predict(scaled)[0]

    future_preds.append(pred)

    # Shift lag values
    current_lags = np.roll(current_lags, 1)

    # Insert the newest prediction as lag_1
    current_lags[0] = pred


# -----------------------
# Prepare future dates
# -----------------------
last_date = df[DATE_COL].iloc[-1]

future_dates = [
    last_date + pd.Timedelta(days=i + 1)
    for i in range(PREDICT_N_DAYS)
]


# -----------------------
# Print future predictions
# -----------------------
print(
    "\nPredictions for next",
    PREDICT_N_DAYS,
    "days:"
)

for d, p in zip(future_dates, future_preds):
    print(f"{d.date()}: {p:.2f}")


# -----------------------
# Residuals calculation
# -----------------------
train_residuals = y_train - y_pred_train

test_residuals = y_test - y_pred_test


# -----------------------
# Plot results
# -----------------------
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    sharex=True
)


# -----------------------
# Top panel:
# Actual vs Predicted
# -----------------------
ax1.plot(
    dates_train,
    y_train,
    label="Train (actual)",
    linewidth=1,
    color="blue",
    alpha=0.7
)

ax1.plot(
    dates_train,
    y_pred_train,
    label="Train (predicted)",
    linestyle="--",
    linewidth=1,
    color="cyan"
)

ax1.plot(
    dates_test,
    y_test,
    label="Test (actual)",
    linewidth=1,
    color="green",
    alpha=0.7
)

ax1.plot(
    dates_test,
    y_pred_test,
    label="Test (predicted)",
    linestyle="--",
    linewidth=1,
    color="orange"
)

ax1.plot(
    future_dates,
    future_preds,
    label="Future predictions",
    marker="o",
    linestyle="-",
    color="red"
)

ax1.set_ylabel("Close Price")

ax1.set_title(
    "AAPL - Time-Series Actual vs. Predicted"
)

ax1.legend()

ax1.grid(
    True,
    linestyle="--",
    alpha=0.5
)


# -----------------------
# Bottom panel:
# Residual errors
# -----------------------
ax2.plot(
    dates_train,
    train_residuals,
    label="Train Residuals",
    linewidth=1,
    color="purple",
    alpha=0.7
)

ax2.plot(
    dates_test,
    test_residuals,
    label="Test Residuals",
    linewidth=1,
    color="red",
    alpha=0.7
)

ax2.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1
)

ax2.set_xlabel("Date")

ax2.set_ylabel(
    "Residual Error (Actual - Pred)"
)

ax2.set_title(
    "Model Residual Errors Over Time"
)

ax2.legend()

ax2.grid(
    True,
    linestyle="--",
    alpha=0.5
)


# -----------------------
# Display plot
# -----------------------
plt.tight_layout()

plt.show()