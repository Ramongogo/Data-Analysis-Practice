# Results 
---
> **The final selected stacking model composed of ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier XGBClassifier and LGBMClassifier has an acurracy of 0.9833.**
---
## Process Elaboration
### 1. Data Processing
* Importing data and required packages
* Checking missing value
* Using heatmap to observe the features' correlation with bankruptcy
### 2. Feature Engineering
* Feature selection - Selcting absolute value of features' correlation higher than 0.1
  ![Features](https://github.com/user-attachments/assets/625e2c78-57c3-429a-bfe3-9186af4cd659)

* Feature extraction - Using PCA (Explained Variance = 95%) to reduce selected features' dimensionality and eliminate collinearity

  ![螢幕擷取畫面 2024-08-01 190701](https://github.com/user-attachments/assets/3dcdb224-d307-4b24-8812-6aa9902ce21f)

  Reducing from 32 features to 13 new features. 
### 3. Handling Imbalanced Dataset
* There is a huge difference between bankruptcy and non-bankruptcy, which is a inbalanced dataset
  ![banruptcy ratio](https://github.com/user-attachments/assets/621edea7-6e1c-42df-ad8f-46b7ace8db18)

* Using SMOTE to randomly create samples of bankruptcy until they are as many as non-bankruptcy's samples
  ![asjusted banruptcy ratio](https://github.com/user-attachments/assets/2d7ecb26-f25e-4c77-999a-e0b18f0cc520)
### 4. Model Training
* Using lazypredict package to automatically run more than 15 models, then selecting the best performing model - ExtraTreeClassifer with accuracy score of 0.9769. 
  ![lazy result](https://github.com/user-attachments/assets/042e98df-ebca-4b6f-af8b-8e2bf943d803)


### 4. Comparison
* The result turned out that Lasso without tuning ranks first, Lasso after tuning ranks second and Linear Regression ranks third. Thus, the features have an obvious linear relationship with revenue. Besides, tuning and stacking aren't effective in this case. 
