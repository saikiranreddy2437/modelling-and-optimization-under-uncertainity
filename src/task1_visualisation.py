import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

math = pd.read_csv('data/student-mat.csv', sep=';')
por  = pd.read_csv('data/student-por.csv', sep=';')
math['subject'] = 'Math'; por['subject'] = 'Portuguese'
df = pd.concat([math, por], ignore_index=True)
df['pass_binary'] = (df['G3'] >= 10).astype(int)

# Figure 1
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x='subject', hue='pass_binary', ax=axes[0])
axes[0].set_title('Pass/Fail by Subject')
axes[1].hist(df['G3'], bins=20, edgecolor='black')
axes[1].axvline(10, color='red', linestyle='--')
axes[1].set_title('G3 Distribution')
sns.boxplot(data=df, x='pass_binary', y='failures', ax=axes[2])
axes[2].set_title('Failures by Pass/Fail')
plt.tight_layout(); plt.savefig('figure1_dataset_overview.png', dpi=300)

# Figure 2
features = ['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'G3']
corr = df[features].corr()
plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0)
plt.title('Feature Correlation with G3')
plt.tight_layout(); plt.savefig('figure2_correlation_heatmap.png', dpi=300)

