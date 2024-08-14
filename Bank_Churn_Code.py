# Checking data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/USER/Downloads/data/BankChurners.csv", encoding = 'utf8')
df = pd.DataFrame(data)
df = df.drop_duplicates()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df.replace('Unknown', np.nan, inplace=True)
df.dropna(inplace=True)
print(df.info())
print(df.head(10))
print(df.nunique())
print(df.duplicated().sum())
print(df.isnull().sum().sum())
print(df['Education_Level'].unique())
print(df['Income_Category'].unique())
print(df['Card_Category'].unique())

# Transform categorical data
from sklearn.preprocessing import LabelEncoder
df = df.drop(['CLIENTNUM'], axis=1)
df = df.drop (columns=df.columns[-2:])
le = LabelEncoder()
df['Attrition_Flag'] = le.fit_transform(df['Attrition_Flag'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Marital_Status'] = le.fit_transform(df['Marital_Status'])
education_map = {'High School': 1,'Graduate': 2,'Uneducated': 0,'College': 2,'Post-Graduate': 3
,'Doctorate': 4}
income_map = {'$60K - $80K': 2,'Less than $40K': 0,'$80K - $120K': 3,'$40K - $60K': 1,'$120K +': 4}
card_map = {'Blue': 0,'Gold': 1,'Silver': 2,'Platinum': 3}
df['Education_Level'] = df['Education_Level'].map(education_map)
df['Income_Category'] = df['Income_Category'].map(income_map)
df['Card_Category'] = df['Card_Category'].map(card_map)
 
# Filtering the data in terms of inactive months more than 2
df_filtered = df[df['Months_Inactive_12_mon']>2]
df_filtered['DV'] = ((df_filtered['Months_Inactive_12_mon'] >=5) & (df['Attrition_Flag'] == 0)).astype(int)
print(df_filtered.info())
print(df_filtered.head())
print(df_filtered['DV'].value_counts())

# Selecting features' absolute value of correlation with revenue higher than 0.08
import seaborn as sns
matrix = df_filtered.corr()
sns.heatmap(matrix, annot=True, cmap='Blues',fmt='.3f')
plt.title('Correlation Matrix')
plt.show()
matrix_dv = matrix['DV'].sort_values(ascending = False)
matrix_dv = matrix_dv.drop(['DV','Months_Inactive_12_mon','Attrition_Flag'])
selected_features = matrix_dv[(matrix_dv >= 0.08) | (matrix_dv <= -0.08)]
print(selected_features)
sns.barplot(x = selected_features.index, y = selected_features.values, palette = 'Set2')
plt.title('Correlation with Bankruptcy')
plt.xlabel('Features')
plt.ylabel('Correlations')
plt.xticks(rotation = 90)
for i, v in enumerate(selected_features) :
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom' if v > 0 else 'top')
plt.show()

# Extracting Features by PCA to eliminate collinearity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
selected_columns = list(selected_features.index)
df_selected_features = df_filtered[selected_columns]
x = df_selected_features
y = df_filtered['DV']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
pca = PCA(n_components = 0.95)
x_pca = pca.fit_transform(x_scaled)
pca_ratio = pca.explained_variance_ratio_.tolist()
pca_name = [f'PCA{i + 1}'for i in range(len(pca_ratio))] 
print(pca_ratio)
sns.barplot(x = pca_name,y = pca_ratio, palette = 'Set2')
plt.title('New Features Correlation')
plt.xlabel('PCA Features')
plt.ylabel('Correlation')
for i ,v in enumerate(pca_ratio):
    plt.text(i, v, f'{v:.2f}', ha = 'center', va = 'top')
plt.show()

# Using SMOTE to handle imbalanced dataset
from imblearn.over_sampling import SMOTE
sns.countplot(x = y, palette = 'Set2')
plt.show()
x_smote, y_smote = SMOTE(sampling_strategy = 0.25,random_state = 88).fit_resample(x_pca, y)
sns.countplot(x = y_smote, palette = 'Set2')
plt.show()

# Using lazypredict to see which models have better performance 
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size = 0.2, random_state= 88)
clf = LazyClassifier(verbose = 1, predictions = True)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print(models)

# Using Optuna to hypertune the best performing model - ExtraTreeClassifier
## Before tuning 
etc = ExtraTreesClassifier(random_state=88)
etc.fit(x_train, y_train)
y_pred = etc.predict(x_test)
cv_scores = cross_val_score(etc, x_train, y_train, cv =10, scoring = 'accuracy', n_jobs = -1)
print(cv_scores, cv_scores.mean())
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot = True, fmt ='d', cmap = 'Oranges')
plt.title('Best Extra Tree Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
## After tuning 
import optuna
def objetive(trial):
    etc2 = ExtraTreesClassifier(
    n_estimators=trial.suggest_int('n_estimators', 50, 300),
    max_depth=trial.suggest_int('max_depth', 5, 50),
    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
    random_state=88)
    cv_scores_best = cross_val_score(etc2, x_train, y_train, cv = 10, scoring = 'accuracy')
    return (cv_scores_best.mean())
study = optuna.create_study(direction='maximize')
study.optimize(objetive, n_trials = 50)
best_params = study.best_trial.params
best_etc = ExtraTreesClassifier(**best_params)
best_etc.fit(x_train, y_train)
best_etc_pred = best_etc.predict(x_test)
cm2 = confusion_matrix(y_test, best_etc_pred)
print(classification_report(y_test, best_etc_pred, digits = 4))
cv_scores_best_etc = cross_val_score(best_etc, x_train, y_train, cv =10, scoring = 'accuracy', n_jobs = -1)
print(cv_scores_best_etc.mean())
sns.heatmap(cm2, annot = True, fmt ='d', cmap = 'Blues')
plt.title('Best Extra Tree Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Forming a stacking model composed of top four performing model
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
meta_clf = LogisticRegression(random_state = 88)
etc = ExtraTreesClassifier(random_state = 88)
rf = RandomForestClassifier(random_state = 88)
xgb = XGBClassifier(random_state = 88)
lgbm = LGBMClassifier(random_state = 88)
stack_clf = StackingClassifier(
    estimators = [('ExtraTrees', etc), ('RandomForest', rf), ('XGB', xgb), ('LGBM', lgbm)],
    final_estimator = meta_clf
    )
stack_clf.fit(x_train, y_train)
stack_pred = stack_clf.predict(x_test)
cm3 = confusion_matrix(y_test, stack_pred)
print(classification_report(y_test, stack_pred, digits = 4))
cv_scores_stack = cross_val_score(stack_clf, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
print(cv_scores_stack.mean())
sns.heatmap(cm3, annot =True, fmt = 'd', cmap = 'Purples')
plt.title('Stacking Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()