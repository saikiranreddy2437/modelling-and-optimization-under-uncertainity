
import numpy as np
import GPy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Load processed data
data = np.load('data/processed_student_data.npz')
X = data['X']
y_reg = data['y_reg']

# Restrict to Portuguese rows (last 649 if you concatenated math then por)
n_por = pd.read_csv('data/student-por.csv', sep=';').shape[0]
X_por = X[-n_por:]
y_por = y_reg[-n_por:]

# Create bins for stratification
bins = pd.cut(y_por, bins=3, labels=False)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rmse_list, mae_list, r2_list = [], [], []

for fold, (tr, te) in enumerate(kf.split(X_por, bins), start=1):
    X_tr, X_te = X_por[tr], X_por[te]
    y_tr, y_te = y_por[tr], y_por[te]

    kernel = GPy.kern.RBF(input_dim=X_por.shape[1], ARD=True) + \
             GPy.kern.White(input_dim=X_por.shape[1])

    m = GPy.models.GPRegression(X_tr, y_tr.reshape(-1, 1), kernel)
    m.optimize(messages=False, max_iters=1000)

    y_pred, _ = m.predict(X_te)
    y_pred = y_pred.flatten()

    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    rmse_list.append(rmse); mae_list.append(mae); r2_list.append(r2)
    print(f"Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")

print("\nGP Regression (Portuguese):")
print(f"RMSE = {np.mean(rse_list):.2f} ± {np.std(rmse_list):.2f}")
print(f"MAE  = {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}")
print(f"R²   = {np.mean(r2_list):.2f} ± {np.std(r2_list):.2f}")


