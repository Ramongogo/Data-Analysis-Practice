import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/USER/Downloads/data/data.csv", encoding = 'utf8')
df = pd.DataFrame(data)
df = df.drop_duplicates()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.duplicated().sum())
print(df.isnull().sum().sum())

matrix = df.corr()
target = 'Bankrupt?'
matrix_bankrupt = matrix[target].sort_values(ascending = False)
matrix_bankrupt = matrix_bankrupt.drop(target)
selected_features = matrix_bankrupt[(matrix_bankrupt >= 0.1) | (matrix_bankrupt <= -0.1)]  
print(selected_features)
selected_columns = list(selected_features.index)
df_selected_features = df[selected_columns]
sns.barplot(x = selected_features.index, y = selected_features.values, palette = 'Set2')
plt.title('Correlation with Bankruptcy')
plt.xlabel('Features')
plt.ylabel('Correlations')
plt.xticks(rotation = 90)
for i, v in enumerate(selected_features) :
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom' if v > 0 else 'top')
plt.show()

from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
x = df_selected_features
y = df['Bankrupt?']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
#kpca = KernelPCA(n_components = 10, kernel = 'rbf')
pca = PCA(n_components = 0.95)
#x_kpca = kpca.fit_transform(x_scaled)
x_pca = pca.fit_transform(x_scaled)
print(pca.explained_variance_ratio_)

from imblearn.over_sampling import SMOTE
sns.countplot(x = y, palette = 'Set2')
plt.show()
x_smote, y_smote = SMOTE().fit_resample(x_pca, y)
sns.countplot(x = y_smote, palette = 'Set2')
plt.show()

from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size = 0.2)
clf = LazyClassifier(verbose = 1, predictions = True)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
print(models)

from sklearn.ensemble import VotingClassifier,StackingClassifier, ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
etc = ExtraTreesClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier()
bag = BaggingClassifier()
lgbm = LGBMClassifier()
vote_clf = VotingClassifier(
    estimators = [('ExtraTrees', etc), ('RandomForest', rf), ('XGB', xgb), ('Bagging', bag), ('LGBM', lgbm)],
    voting = 'soft'
    )
vote_clf.fit(x_train, y_train)
vote_pred = vote_clf.predict(x_test)
print(classification_report(y_test, vote_pred))
cv_scores_vote = cross_val_score(vote_clf, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
print(cv_scores_vote)
cm = confusion_matrix(y_test, vote_pred)
sns.heatmap(cm, annot =True, fmt = 'd', cmap = 'Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

meta_clf = LogisticRegression()
stack_clf = StackingClassifier(
    estimators = [('ExtraTrees', etc), ('RandomForest', rf), ('XGB', xgb), ('Bagging', bag), ('LGBM', lgbm)],
    final_estimator = meta_clf
)
stack_clf.fit(x_train, y_train)
stack_pred = stack_clf.predict(x_test)
cm2 = confusion_matrix(y_test, stack_pred)
print(classification_report(y_test, stack_pred))
cv_scores_stack = cross_val_score(stack_clf, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
print(cv_scores_vote)
sns.heatmap(cm2, annot =True, fmt = 'd', cmap = 'Purples')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()