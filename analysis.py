import pandas as pd
import numpy as np
from models import *
from sklearn.metrics import classification_report

# Logistic Regression importance
log_importance = pd.Series(
    np.abs(log_model.coef_[0]),
    index=X_train.columns
).sort_values(ascending=False)

print("\nLogistic Regression Feature Importance:")
print(log_importance)


# Random Forest importance
rf_importance = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print("\nRandom Forest Feature Importance:")
print(rf_importance)


# Find top features
print("\nTop feature (Logistic Regression):", log_importance.idxmax())
print("Top feature (Random Forest):", rf_importance.idxmax())

# Additionl Metrics
print("\nLogistic Regression Report:")
print(classification_report(y_val, y_val_pred_log))

print("\nRandom Forest Report:")
print(classification_report(y_val, y_val_pred_rf))