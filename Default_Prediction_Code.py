# Checking Data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/USER/Downloads/data/Loan_Default.csv", encoding = 'utf8')
df = pd.DataFrame(data)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.info())
print(df.head(10))
print(df.nunique())

# Transforming categorical data
from sklearn.preprocessing import LabelEncoder
df = df.drop(['ID', 'year', 'open_credit','construction_type', 'Secured_by', 'Security_Type'], axis = 1)
le = LabelEncoder()
list_le = ["loan_limit",'approv_in_adv','Credit_Worthiness','business_or_commercial','Neg_ammortization',
           'interest_only','lump_sum_payment','co-applicant_credit_type','submission_of_application']
for i in list_le:
    df[i] = le.fit_transform(df[i])

df["Gender"] = df["Gender"].replace(['Sex Not Available'], np.nan)
df.dropna(subset = ['Gender'], inplace=True)
list_dummy = ['Gender','loan_type','loan_purpose','occupancy_type','credit_type','Region']
dummy_dict = {}
for i in list_dummy:
    dummy_dict[f'df_{i}'] = pd.get_dummies(df[i],drop_first=False).astype(int)
    df = pd.concat([df,dummy_dict[f'df_{i}']], axis=1)
    df = df.drop([i],axis=1)

map_units = {'1U':1,'2U':2,'3U':3,'4U':4}
df["total_units"] = df["total_units"].map(map_units)
map_age = {'<25':0,'25-34':1,'35-44':2,'45-54':3,'55-64':4,'65-74':5,'>74':6}
df["age"] = df["age"].map(map_age)

list_fillna = ['rate_of_interest','Interest_rate_spread','Upfront_charges','term','property_value','income','LTV','dtir1']
for i in list_fillna:
    df[i] = df[i].fillna(df[i].mean())

# Selecting features' absolute value of correlation with revenue higher than 0.08
matrix = df.corr()
matrix_status = matrix['Status'].sort_values(ascending = False)
matrix_status = matrix_status.drop(['Status'])
selected_features = matrix_status[(matrix_status >= 0.08) | (matrix_status <= -0.08)]
print(selected_features)
sns.barplot(x = selected_features.index, y = selected_features.values, palette = 'Set2')
plt.title('Correlation With Default Status')
plt.xlabel('Features')
plt.ylabel('Correlations')
plt.xticks(rotation = 90)
for i, v in enumerate(selected_features) :
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom' if v > 0 else 'top')
plt.show()

selected_columns = list(selected_features.index)
df_selected_features = df[selected_columns] 
df_selected_features = pd.concat([df_selected_features, df['Status']], axis=1)
for i in range(len(df_selected_features.columns)-1):
    Q1 = np.percentile(df_selected_features.iloc[:,i], 25, method = 'midpoint')
    Q3 = np.percentile(df_selected_features.iloc[:,i], 75, method = 'midpoint')
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    outliers = (df_selected_features.iloc[:,i] > upper) | (df_selected_features.iloc[:,i] < lower)
df_selected_features = df_selected_features[~outliers]    
print(df_selected_features.info())

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = df_selected_features.drop(['Status'], axis=1)
y = df_selected_features['Status']
x_scaled = scaler.fit_transform(x)
pca = PCA(n_components = 0.95)
x_pca = pca.fit_transform(x_scaled)
pca_ratio = pca.explained_variance_ratio_
pca_name = [f'PCA{i + 1}'for i in range(len(pca_ratio))] 
sns.barplot(x = pca_name,y = pca_ratio, palette = 'Set2')
plt.title('New Features Correlation')
plt.xlabel('PCA Features')
plt.ylabel('Correlation')
for i ,v in enumerate(pca_ratio):
    plt.text(i, v, f'{v:.2f}', ha = 'center', va = 'top')
plt.show()

from imblearn.over_sampling import SMOTE
sns.countplot(x = y, palette = 'Set2')
plt.show()
x, y = SMOTE(sampling_strategy = 0.4,random_state = 88).fit_resample(x_pca, y)
sns.countplot(x = y, palette = 'Set2')
plt.show()

from catboost import CatBoostClassifier 
from xgboost import XGBClassifier 
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 88)
models = {'CatBoostClassifier':CatBoostClassifier(),'XGBClassifier':XGBClassifier(), 
          'RandomForestClassifier':RandomForestClassifier(),'SVC':SVC(),
          'GaussianNB':GaussianNB(),"LogisticRegression":LogisticRegression(),
          'KNeighborsClassifier':KNeighborsClassifier(),'LGBMClassifier':LGBMClassifier()}
results = {}
for model_name, model in models.items():
    model.fit(x_train,y_train)
    y_test_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    results[model_name] = {'accuracy_score':accuracy,'f1_score':f1}
#results = {'CatBoostClassifier': {'accuracy_score': 0.8379942548158161, 'f1_score': 0.6067070044098041}, 'XGBClassifier': {'accuracy_score': 0.8379942548158161, 'f1_score': 0.6067070044098041}, 'RandomForestClassifier': {'accuracy_score': 0.8379942548158161, 'f1_score': 0.6067070044098041}, 'SVC': {'accuracy_score': 0.8380364988171679, 'f1_score': 0.6067692307692307}, 'GaussianNB': {'accuracy_score': 0.8176326461642447, 'f1_score': 0.6119550561797753}, 'LogisticRegression': {'accuracy_score': 0.8379942548158161, 'f1_score': 0.605898674339739}, 'KNeighborsClassifier': {'accuracy_score': 0.8170834741466712, 'f1_score': 0.5980319346453769}, 'LGBMClassifier': {'accuracy_score': 0.8380364988171679, 'f1_score': 0.6068498769483183}}
df_result = pd.DataFrame(results).T.reset_index()
df_result.columns = ['Model', 'Accuracy', 'F1_Score']
df_result = df_result.sort_values(by='F1_Score', ascending=False)
print(df_result)
gaussian = GaussianNB()
gaussian.fit(x_train,y_train)
gaussian_test_pred = model.predict(x_test)
#cv_scores_stack = cross_val_score(gaussian, x_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
cm = confusion_matrix(y_test, gaussian_test_pred)
print(classification_report(y_test, gaussian_test_pred, digits = 4))
#print(cv_scores_stack.mean())
sns.heatmap(cm, annot =True, fmt = 'd', cmap = 'Blues')
plt.title('Gaussian Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

meta_clf = LogisticRegression(random_state = 88)
lgr = LogisticRegression(random_state = 88)
svc = SVC(random_state = 88)
xgb = XGBClassifier(random_state = 88)
gau = GaussianNB()
stack_clf = StackingClassifier(
    estimators = [('LogisticRegression', lgr), ('SVC', svc), ('XGB', xgb), ('GaussianNB', gau)],
    final_estimator = meta_clf
    )
stack_clf.fit(x_train, y_train)
stack_pred = stack_clf.predict(x_test)
cm2 = confusion_matrix(y_test, stack_pred)
print(classification_report(y_test, stack_pred, digits = 4))
sns.heatmap(cm2, annot =True, fmt = 'd', cmap = 'Purples')
plt.title('Stacking Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
