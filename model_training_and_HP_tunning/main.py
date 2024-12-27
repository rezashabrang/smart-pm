import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV


# Step 1: Generate synthetic data for project management
# Define a function to create synthetic data with relevant features
def generate_synthetic_data(num_samples=1000):
    """
    Generate synthetic project management data with relevant features.

    Features:
    - Team Size: Number of team members.
    - Resource Cost: Total cost of resources allocated.
    - Predicted Cost: Estimated cost based on project planning tools.
    - Duration: Estimated duration of the project (days).
    - Complexity: A measure of project complexity (scale 1-10).
    - Risk Factor: A measure of potential project risks (scale 1-10).

    Target:
    - Actual Cost: The actual cost incurred for the project.

    Returns:
    - DataFrame with features and target variable.
    """
    np.random.seed(42)

    # Features
    team_size = np.random.randint(5, 50, num_samples)
    resource_cost = np.random.uniform(1000, 50000, num_samples)
    predicted_cost = resource_cost * (1 + np.random.uniform(-0.2, 0.2, num_samples))
    duration = np.random.randint(30, 365, num_samples)
    complexity = np.random.randint(1, 11, num_samples)
    risk_factor = np.random.randint(1, 11, num_samples)

    # Target variable: Actual cost
    actual_cost = (
        predicted_cost
        + np.random.normal(0, 5000, num_samples)
        + complexity * 1000
        - risk_factor * 500
    )

    data = pd.DataFrame(
        {
            "Team Size": team_size,
            "Resource Cost": resource_cost,
            "Predicted Cost": predicted_cost,
            "Duration": duration,
            "Complexity": complexity,
            "Risk Factor": risk_factor,
            "Actual Cost": actual_cost,
        }
    )

    return data


# Generate the synthetic data
data = generate_synthetic_data()

# Step 2: Split the data into training and testing sets
X = data.drop(columns=["Actual Cost"])
y = data["Actual Cost"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Define models to evaluate
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", random_state=42),
    "Neural Network": MLPRegressor(random_state=42, max_iter=500),
}


# Step 4: Define hyperparameter tuning methods and their configurations
def perform_tuning(model, param_grid, X_train, y_train, tuning_method="grid"):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

    Args:
    - model: Machine learning model to tune.
    - param_grid: Dictionary of hyperparameters to tune.
    - X_train: Training features.
    - y_train: Training target.
    - tuning_method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV.

    Returns:
    - Best model after tuning.
    - Best parameters.
    """
    if tuning_method == "grid":
        tuner = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            verbose=1,
        )
    elif tuning_method == "random":
        tuner = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring="neg_mean_squared_error",
            random_state=42,
            verbose=1,
        )
    else:
        raise ValueError("Invalid tuning method. Choose 'grid' or 'random'.")

    tuner.fit(X_train, y_train)
    return tuner.best_estimator_, tuner.best_params_


# Step 5: Perform hyperparameter tuning for each model
# Example: Hyperparameter tuning for Random Forest
rf_param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

best_rf_model, best_rf_params = perform_tuning(
    models["Random Forest"], rf_param_grid, X_train, y_train, tuning_method="grid"
)

# Example: Hyperparameter tuning for XGBoost
xgb_param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 6, 9],
}

best_xgb_model, best_xgb_params = perform_tuning(
    models["XGBoost"], xgb_param_grid, X_train, y_train, tuning_method="random"
)


# Step 6: Evaluate models after tuning
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test data.

    Args:
    - model: Trained model.
    - X_test: Test features.
    - y_test: Test target.

    Returns:
    - Mean Squared Error.
    - R² Score.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2


rf_mse, rf_r2 = evaluate_model(best_rf_model, X_test, y_test)
xgb_mse, xgb_r2 = evaluate_model(best_xgb_model, X_test, y_test)

# Step 7: Output results
print("Random Forest Best Parameters:", best_rf_params)
print("Random Forest Test MSE:", rf_mse)
print("Random Forest Test R²:", rf_r2)

print("XGBoost Best Parameters:", best_xgb_params)
print("XGBoost Test MSE:", xgb_mse)
print("XGBoost Test R²:", xgb_r2)
