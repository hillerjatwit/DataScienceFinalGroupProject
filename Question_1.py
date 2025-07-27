import pandas as pd
import fastf1
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# get session info for each driver
def GetDriverData(track, driver, year):

    return_value = []

    # Get all the laps of the driver for the session
    session = fastf1.get_session(year, track, 'R')
    session.load()

    weather_data = session.weather_data.copy()
    weather_data = weather_data.sort_values('Time')

    laps = session.laps.pick_driver(driver)
    #laps = laps[~laps['IsPitOutLap'] & ~laps['IsPitInLap']]
    #laps = laps[~laps['LapTime'].isna()]
    laps.reset_index(drop=True, inplace=True)

    all_lap_data = []

    for _, lap in laps.iterlaps():

        telemetry = lap.get_telemetry()
        car_data = lap.get_car_data()

        merged = pd.merge_asof(
            telemetry.sort_values('Time'),
            car_data.sort_values('Time'),
            on='Time',
            direction='nearest',
            tolerance=pd.Timedelta(milliseconds=5)
        )

        merged = pd.merge_asof(
            merged.sort_values('Time'),
            weather_data.sort_values('Time'),
            on='Time',
            direction='nearest',
            tolerance=pd.Timedelta(seconds=10)  # Weather data is lower resolution
        )

        merged.dropna(inplace=True)

        merged['LapNumber'] = lap['LapNumber']
        merged['LapTime'] = lap['LapTime'].total_seconds()
        merged['Driver'] = lap['Driver']
        all_lap_data.append(merged)

    return_value = pd.concat(all_lap_data, ignore_index=True)

    return return_value

def preprocess(df, fit_scaler=True, scaler=None):
    df = df.dropna()
    #df['DRS'] = df['DRS'].astype(int)
    
    features = [
        "Distance", "Speed_x", "Throttle_x", "Brake_x", "nGear_x", "RPM_x", "DRS_x",
        "AirTemp", "TrackTemp", "Humidity", "Pressure", "Rainfall"
    ]

    X = df[features]
    y = df["LapTime"]

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y.values, scaler
    else:
        X_scaled = scaler.transform(X)
        return X_scaled, y.values, scaler


training_data = pd.concat([
    GetDriverData("Monza", "VER", 2021),
    GetDriverData("Monza", "VER", 2022),
    GetDriverData("Monza", "VER", 2023)
], ignore_index=True)
test_data = GetDriverData("Monza", "VER", 2024)

X_train, y_train, scaler = preprocess(training_data, fit_scaler=True)
X_test, y_test, _ = preprocess(test_data, fit_scaler=False, scaler=scaler)

models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(n_neighbors=5),
}

results_df = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    percent_errors = 100 * (y_pred - y_test) / y_test
    mean_percent_error = np.mean(np.abs(percent_errors))
    variance = np.var(y_pred - y_test)

    results_df.append({
        "Model": name,
        "MAE": mae,
        "R²": r2,
        "Mean % Error": mean_percent_error,
        "Variance of Error": variance
    })

    model_df = pd.DataFrame({
        "Actual LapTime": y_test,
        "Predicted LapTime": y_pred,
        "Percent Error": percent_errors
    })
    model_df["Model"] = name
    results_df.append(model_df)

# Combine summary results and detailed model predictions
summary_df = pd.DataFrame([r for r in results_df if isinstance(r, dict)])
detailed_df = pd.concat([r for r in results_df if isinstance(r, pd.DataFrame)], ignore_index=True)

# Set up Seaborn style
sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(14, 8))

# Plot MAE per model
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x="Model", y="MAE", palette="Blues_d")
plt.title("Mean Absolute Error by Model")
plt.ylabel("MAE (seconds)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Plot R² Score per model
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x="Model", y="R²", palette="Greens_d")
plt.title("R² Score by Model")
plt.ylabel("R²")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Plot Mean Percent Error per model
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x="Model", y="Mean % Error", palette="Oranges_d")
plt.title("Mean Percent Error by Model")
plt.ylabel("Percent Error (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Plot Error Variance per model
plt.figure(figsize=(10, 6))
sns.barplot(data=summary_df, x="Model", y="Variance of Error", palette="Purples_d")
plt.title("Variance of Error by Model")
plt.ylabel("Variance")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Plot Actual vs Predicted Lap Time
plt.figure(figsize=(14, 8))
sns.scatterplot(data=detailed_df, x="Actual LapTime", y="Predicted LapTime", hue="Model", alpha=0.6)
plt.plot([detailed_df["Actual LapTime"].min(), detailed_df["Actual LapTime"].max()],
         [detailed_df["Actual LapTime"].min(), detailed_df["Actual LapTime"].max()],
         'k--', label="Ideal Prediction")
plt.title("Actual vs Predicted Lap Times")
plt.xlabel("Actual Lap Time (s)")
plt.ylabel("Predicted Lap Time (s)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot percent error distributions
plt.figure(figsize=(14, 8))
sns.boxplot(data=detailed_df, x="Model", y="Percent Error", palette="coolwarm")
plt.title("Distribution of Percent Errors by Model")
plt.ylabel("Percent Error (%)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()