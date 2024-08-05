## Results 
---
> **The final selected stacking model composed of ExtraTreesClassifier, RandomForestClassifier, BaggingClassifier XGBClassifier and LGBMClassifier has an acurracy of 0.9833.**
---
## Process Elaboration
### 1. Data Processing
* Importing data and required packages
* Checking missing value
* Using heatmap to observe the features' correlation with bankruptcy
### 2. Feature Engineering
* Feature selection 
  * Selcting absolute value of features' correlation higher than 0.1
  ![Features](https://github.com/user-attachments/assets/625e2c78-57c3-429a-bfe3-9186af4cd659)

* Feature extraction 
  * Using PCA (Explained Variance = 95%) to reduce selected features' dimensionality and eliminate collinearity
    ![new featurs](https://github.com/user-attachments/assets/56e013c4-8795-42d9-b0ca-bd05f4978567)
  * Reducing from 32 features to 13 new features. 
### 3. Handling Imbalanced Dataset
* There is a huge difference between bankruptcy and non-bankruptcy, which is a inbalanced dataset
  ![banruptcy ratio](https://github.com/user-attachments/assets/621edea7-6e1c-42df-ad8f-46b7ace8db18)

* Using SMOTE to randomly create samples of bankruptcy until they are as many as non-bankruptcy's samples
  ![asjusted banruptcy ratio](https://github.com/user-attachments/assets/2d7ecb26-f25e-4c77-999a-e0b18f0cc520)
### 4. Model Training
* Using lazypredict package to automatically run more than 15 models, then selecting the best performing model - ExtraTreeClassifer with accuracy score of 0.9769.
  
  ![螢幕擷取畫面 2024-08-01 230623](https://github.com/user-attachments/assets/c3ac0557-eadb-497e-8a0a-81f19fd99ade)

### 5. Model Tuning
* Using Optuna to hypertune the best performing model - ExtraTreeClassifier

  ![螢幕擷取畫面 2024-08-02 022009](https://github.com/user-attachments/assets/eac4c990-7027-4b55-8669-0d8703cd0b8a)
* There is no difference between with the result of lazypredict, which already has the best parameters.
  #### Tuned ExtraTreeClassifier's Confusion Matrix
  ![extra result](https://github.com/user-attachments/assets/665da790-3e44-4b3f-a4f0-dd0a23e4fdad)
### 6. Model Voting
* Forming a soft voting model composed of top five performing model
  * VotingClassifier
  * StackingClassifier
  * ExtraTreesClassifier
  * RandomForestClassifier
  * BaggingClassifier
    
  ![螢幕擷取畫面 2024-08-02 022348](https://github.com/user-attachments/assets/e5aa0bf0-fabd-44f4-9692-09d6f12a6164)
* Performance of voting model with accuracy score of 0.9697 is slightly worse than ExtraTreeClassifier
  #### Voting Model's Confusion Matrix
  ![vote result](https://github.com/user-attachments/assets/4046ca25-a334-4645-a6e5-ab97caed6558)
### 7. Model Stacking ✔️
* Forming a stacking model composed of top five performing model
  
  ![螢幕擷取畫面 2024-08-02 022331](https://github.com/user-attachments/assets/eeea6108-2cce-47ef-8e99-07ccebf2f971)
* Performance of stacking model with accuracy score of 0.9833 is slightly better than ExtraTreeClassifier
  #### Stacking Model's Confusion Matrix
  ![stack result](https://github.com/user-attachments/assets/446d808f-c089-4a12-9877-33a1a1fe3ace)

