import numpy as np
import GPy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import pandas as pd

data = np.load('data/processed_student_data.npz')
X = data['X']
y_cls = data['y_cls']

n_por = pd.read_csv('data/student-por.csv', sep=';').shape[0]
X_por = X[-n_por:]
y_por = y_cls[-n_por:]

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_list, acc_list, ll_list = [], [], []

for fold, (tr, te) in enumerate(kf.split(X_por, y_por), start=1):
    X_tr, X_te = X_por[tr], X_por[te]
    y_tr, y_te = y_por[tr], y_por[te]

    kernel = GPy.kern.RBF(input_dim=X_por.shape[1], ARD=True)
    m = GPy.models.GPClassification(X_tr, y_tr.reshape(-1, 1), kernel=kernel)
    m.optimize(messages=False, max_iters=1000)

    y_prob, _ = m.predict(X_te)
    y_prob = y_prob.flatten()
    y_hat = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_te, y_prob)
    acc = accuracy_score(y_te, y_hat)
    ll  = log_loss(y_te, y_prob)

    auc_list.append(auc); acc_list.append(acc); ll_list.append(ll)
    print(f"Fold {fold}: AUC={auc:.2f}, Acc={acc*100:.1f}%, LogLoss={ll:.2f}")

print("\nGP Classification (Portuguese, pass/fail):")
print(f"AUC      = {np.mean(auc_list):.2f} ± {np.std(auc_list):.2f}")
print(f"Accuracy = {np.mean(acc_list)*100:.1f}% ± {np.std(acc_list)*100:.1f}%")
print(f"LogLoss  = {np.mean(ll_list):.2f} ± {np.std(ll_list):.2f}")

