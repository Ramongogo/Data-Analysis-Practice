# Results 
---
> **The final selected stacking model composed of ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier XGBClassifier and LGBMClassifier has an acurracy of 0.9833.**
---
## Motive
Out of my personal interest of exploring delicious foods and decent restaurants, I wondered what kind of features would influence a restaurant's business. Then I found this dataset on Kaggle which has enough samples and features, so I tried doing the analysis by machine learning models. 
## Process Elaboration
### 1. Data Processing
* Importing data and required packages
* Checking missing value
* Using dummmy function to tranform the nominal data column "Cuisine_Type", then Observing correlation between each columns by heatmap
* Removing outliers
* Using heatmap to observe the features' correlation with revenue.
![螢幕擷取畫面 2024-07-26 112607](https://github.com/user-attachments/assets/fc3a21fc-e818-415b-8156-c52b0c2f5103)
### 2. Model Training and Tuning 
* Features Selection
  * The outcome of model trained with these three features has the highest R2 score.
* Using Linear Regression, Lasso, Ridge, Decision Tree Regressor, Random Forest Regressor, XGB Regressor, Catboosting Regressor and Adaboost Regressor to figure out the best model.
* Tuning
  * At first, Using GridSearchCV method to tune hyperparameter of each model. The result turned out to be the same as the original. Thus, using Optuna method to tune the lasso model which has the highest R2 score. However, the R2 score declined slightly after tuning.
* Cross validating models
### 3. Model Stacking
* Stacking 3 different types of regression model with higher R2 score, including Lasso, Random Forest Regressor and Adaboost Regressor. However, the model's performance is not better. 
### 4. Comparison
* The result turned out that Lasso without tuning ranks first, Lasso after tuning ranks second and Linear Regression ranks third. Thus, the features have an obvious linear relationship with revenue. Besides, tuning and stacking aren't effective in this case. 
