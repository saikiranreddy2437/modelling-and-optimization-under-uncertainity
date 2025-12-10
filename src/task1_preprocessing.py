
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load UCI data (place CSVs in data/ folder)
math = pd.read_csv('data/student-mat.csv', sep=';')
por = pd.read_csv('data/student-por.csv', sep=';')

math['subject'] = 'Math'
por['subject'] = 'Portuguese'
df = pd.concat([math, por], ignore_index=True)

# Targets
y_reg = df['G3'].values                             # regression
df['pass_binary'] = (df['G3'] >= 10).astype(int)    # classification
y_cls = df['pass_binary'].values

# Features (no G1, G2)
numeric_features = ['studytime', 'failures', 'absences', 'Medu', 'Fedu']
categorical_features = ['schoolsup', 'famsup', 'sex', 'address']

# One‑hot encode categoricals
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_cat = encoder.fit_transform(df[categorical_features])
cat_cols = encoder.get_feature_names_out(categorical_features)
df_cat = pd.DataFrame(X_cat, columns=cat_cols, index=df.index)

# Combine numeric + categorical
X_num = df[numeric_features]
X_raw = pd.concat([X_num, df_cat], axis=1)

# Standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Save for other scripts
np.savez('data/processed_student_data.npz',
         X=X_scaled, y_reg=y_reg, y_cls=y_cls)

print("Preprocessing done. X shape:", X_scaled.shape)


