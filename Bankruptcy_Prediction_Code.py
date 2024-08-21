# Cleaning data
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

# Observing correlation between features 
# Selecting absolute value of features' correlation higher than 0.1
matrix = df.corr()
target = 'Bankrupt?'
matrix_bankrupt = matrix[target].sort_values(ascending = False)
matrix_bankrupt = matrix_bankrupt.drop(target)
selected_features = matrix_bankrupt[(matrix_bankrupt >= 0.1) | (matrix_bankrupt <= -0.1)]  
print(selected_features)
sns.barplot(x = selected_features.index, y = selected_features.values, palette = 'Set2')
plt.title('Correlation with Bankruptcy')
plt.xlabel('Features')
plt.ylabel('Correlations')
plt.xticks(rotation = 90)
for i, v in enumerate(selected_features) :
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom' if v > 0 else 'top')
plt.show()

# extracting Features by PCA to eliminate collinearity
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
selected_columns = list(selected_features.index)
df_selected_features = df[selected_columns]
x = df_selected_features
y = df['Bankrupt?']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
#kpca = KernelPCA(n_components = 10, kernel = 'rbf')
pca = PCA(n_components = 0.95)
#x_kpca = kpca.fit_transform(x_scaled)
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


# Using SMOTE to deal with  imbalanced dataset
from imblearn.over_sampling import SMOTE
sns.countplot(x = y, palette = 'Set2')
plt.show()
x_smote, y_smote = SMOTE(random_state = 88).fit_resample(x_pca, y)
sns.countplot(x = y_smote, palette = 'Set2')
plt.show()

# Using lazypredict to see which models have better performance 
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size = 0.2, random_state= 88)
clf = LazyClassifier(verbose = 1, predictions = True)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print(models)

# Using Optuna to hypertune the best performing model - ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
import optuna
from sklearn.metrics import classification_report, confusion_matrix
def objective(trial):
    etc2 = ExtraTreesClassifier(
    n_estimators=trial.suggest_int('n_estimators', 50, 300),
    max_depth=trial.suggest_int('max_depth', 5, 50),
    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
    random_state=88)
    cv_scores_best = cross_val_score(etc2, x_train, y_train, cv = 10, scoring = 'accuracy')
    return np.mean(cv_scores_best)
study = optuna.create_study(direction = 'maximize')
study.optimize(objective, n_trials = 50)
best_params = study.best_trial.params
best_etc = ExtraTreesClassifier(**best_params)
best_etc.fit(x_train, y_train)
best_etc_pred = best_etc.predict(x_test)
cm3 = confusion_matrix(y_test, best_etc_pred)
print(classification_report(y_test, best_etc_pred, digits = 4))
cv_scores_best_etc = cross_val_score(best_etc, x_train, y_train, cv =10, scoring = 'accuracy', n_jobs = -1)
print(cv_scores_best_etc.mean())
sns.heatmap(cm3, annot = True, fmt ='d', cmap = 'Oranges')
plt.title('Best Extra Tree Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Forming a voting model composed of top five performing model
from sklearn.ensemble import VotingClassifier,StackingClassifier, ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
etc = ExtraTreesClassifier(random_state = 88)
rf = RandomForestClassifier(random_state = 88)
xgb = XGBClassifier(random_state = 88)
bag = BaggingClassifier(random_state = 88)
lgbm = LGBMClassifier(random_state = 88)
vote_clf = VotingClassifier(
    estimators = [('ExtraTrees', etc), ('RandomForest', rf), ('XGB', xgb), ('Bagging', bag), ('LGBM', lgbm)],
    voting = 'soft'
    )
vote_clf.fit(x_train, y_train)
vote_pred = vote_clf.predict(x_test)
cm = confusion_matrix(y_test, vote_pred)
print(classification_report(y_test, vote_pred, digits = 4))
cv_scores_vote = cross_val_score(vote_clf, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
print(cv_scores_vote.mean())
sns.heatmap(cm, annot =True, fmt = 'd', cmap = 'Blues')
plt.title('Voting Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Forming a stacking model composed of top five performing model
meta_clf = LogisticRegression(random_state = 88)
stack_clf = StackingClassifier(
    estimators = [('ExtraTrees', etc), ('RandomForest', rf), ('XGB', xgb), ('Bagging', bag), ('LGBM', lgbm)],
    final_estimator = meta_clf
)
stack_clf.fit(x_train, y_train)
stack_pred = stack_clf.predict(x_test)
cm2 = confusion_matrix(y_test, stack_pred)
print(classification_report(y_test, stack_pred, digits = 4))
cv_scores_stack = cross_val_score(stack_clf, x_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
print(cv_scores_stack.mean())
sns.heatmap(cm2, annot =True, fmt = 'd', cmap = 'Purples')
plt.title('Stacking Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()