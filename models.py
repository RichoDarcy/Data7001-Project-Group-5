from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_loading import *

# Logistic Regression
log_model = LogisticRegression(max_iter=10000)
log_model.fit(X_train, y_train)

y_val_pred_log = log_model.predict(X_val)
print("Logistic Regression Validation Accuracy:",
      accuracy_score(y_val, y_val_pred_log))


# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_val_pred_rf = rf_model.predict(X_val)
print("Random Forest Validation Accuracy:",
      accuracy_score(y_val, y_val_pred_rf))