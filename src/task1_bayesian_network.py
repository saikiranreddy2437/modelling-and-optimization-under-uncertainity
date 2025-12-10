import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/student-por.csv', sep=';')
df['pass_binary'] = (df['G3'] >= 10).astype(int)

bn_vars = ['sex', 'Medu', 'studytime', 'failures',
           'schoolsup', 'famsup', 'absences', 'pass_binary']
df_bn = df[bn_vars].copy()

df_bn['Medu']      = pd.cut(df_bn['Medu'],      bins=[-1,0,2,4],      labels=['low','mid','high'])
df_bn['studytime'] = pd.cut(df_bn['studytime'], bins=[0,1,3,4],       labels=['low','mid','high'])
df_bn['absences']  = pd.cut(df_bn['absences'],  bins=[-1,5,15,100],   labels=['low','mid','high'])

train_df, test_df = train_test_split(df_bn, test_size=0.2,
                                     stratify=df_bn['pass_binary'],
                                     random_state=42)

bic = BicScore(train_df)
best_model = bic.best_model()
print("Learned edges:", best_model.edges())

model = BayesianNetwork(best_model.edges())
model.fit(train_df, estimator=MaximumLikelihoodEstimator)
infer = VariableElimination(model)

# Example query
q = infer.query(variables=['pass_binary'],
                evidence={'studytime': 'high', 'absences': 'high'})
print("P(pass | high studytime, high absences) =", q.values[1])

# Simple accuracy
y_true = test_df['pass_binary'].values
y_pred = []
for _, row in test_df.iterrows():
    ev = row.drop('pass_binary').to_dict()
    q = infer.query(variables=['pass_binary'], evidence=ev)
    y_pred.append(int(q.values[1] >= 0.5))

print("BN accuracy:", accuracy_score(y_true, y_pred))

