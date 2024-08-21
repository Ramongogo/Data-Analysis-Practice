## Results 
---
>**The final selected stacking model composed of ExtraTreesClassifier, RandomForestClassifier, XGBClassifier and LGBMClassifier has a weighted average f1 score of 0.9862, capturing 93% of customers whose credit cards have been inactive for two months and are likely to terminate the bank's credit card contract due to prolonged inactivity ,and estimating not only to save 55% of management cost of inactive credit cards but also reduce capital requirement.**
---
## Motive 
---
>**With an active card rate of only 63.8% for credit cards across Taiwan, the high number of inactive cards may result in a significant management cost. Data indicates that on average, a bank in Taiwan spends billions on inactive cards. Additionally, there is a potential risk of fraud and unauthorized transactions. Therefore, based on the available dataset on Kaggle, I design a model that can predict customers who will stop using their cards due to prolonged inactivity.**
---
## Process Elaboration
### 1. Data Processing
* Importing data 
* Checking missing and duplicate data
* Using map and labelencoder fucntion to transform categorical data
* Filtering the data in terms of inactive months more than 2
* Using heatmap to observe the features' correlation with bankruptcy
  ![revenue matrix](https://github.com/user-attachments/assets/3676ca8d-b765-4ebf-86be-b43631bb7ef1)
### 2. Feature Engineering
* Feature selection 
  * Selcting absolute value of features' correlation higher than 0.08
  ![project 1](https://github.com/user-attachments/assets/cf1bbb00-ba8c-4760-b00a-44254af6896a)

* Feature extraction 
  * Using PCA (Explained Variance = 95%) to reduce selected features' dimensionality and eliminate collinearity
  ![project 2](https://github.com/user-attachments/assets/3c30ba79-78f6-47fa-8914-3d7f195a7aee)
### 3. Handling Imbalanced Dataset
* There is a huge difference between the variable, which is a inbalanced dataset
  ![project 3](https://github.com/user-attachments/assets/f6134167-a26e-409b-b564-6f630cbadd76)

* Using SMOTE to randomly create samples (25% of value zero )
  ![project 4](https://github.com/user-attachments/assets/75102df1-8f4b-4521-8b86-c1bae12a8b49)
### 4. Model Training
* Using lazypredict package to automatically run more than 15 models, then selecting the best performing model - ExtraTreeClassifer with f1 score of 0.98.
  
  ![project 8](https://github.com/user-attachments/assets/d358140f-e5de-4522-933c-83b26fcd1610)
  #### ExtraTreeClassifier's Confusion Matrix
  ![project 5](https://github.com/user-attachments/assets/beda88e9-9962-4d5d-aa41-51c03803a22b)
* The model's mean score of 10 times cross validation is 0.9796.
### 5. Model Tuning
* Using Optuna to hypertune the best performing model - ExtraTreeClassifier

  ![螢幕擷取畫面 2024-08-15 230817](https://github.com/user-attachments/assets/26e06e8f-605a-4b95-bdb5-d05f87bf812d)
* There is no big difference between with the f1 score of original ExtraTreeClassifier.
  #### Tuned ExtraTreeClassifier's Confusion Matrix
  ![project 10](https://github.com/user-attachments/assets/94aa477c-9d12-4f62-8e1a-cb544e5080cf)
* The model's mean score of 10 times cross validation is 0.9798, which is slightly better.
### 6. Model Stacking ✔️
* Forming a stacking model composed of top five performing model
  
  ![螢幕擷取畫面 2024-08-15 225947](https://github.com/user-attachments/assets/14dad28c-2ad5-4db1-9f02-490ba1c29284)
* Performance of stacking model with a weighted average f1 score of 0.9862 is slightly better than ExtraTreeClassifier.
  #### Stacking Model's Confusion Matrix
  ![project 7](https://github.com/user-attachments/assets/9c528c01-88d7-4b04-88b3-12cd8c498d78)
* The model's mean score of 10 times cross validation is 0.9805, which is also slightly better than ExtraTreeClassifier after tuning.
## Estimated Benefits 
---
>**The final selected stocking model has a weighted average f1 score of 0.9862 and 10 times cross validations' mean score of 0.9805, which means the model is quite accurate and robust. According to its confusion matrix, we can assume that the model can accurately capture and predict  93% of customers whose cards have been inactive for two months and are likely to terminate the bank's credit card contract due to prolonged inactivity.(152/152+5+6=93%). Besides, the model is estimated to be able to also save the 55% of management cost of those inactive cards because it predicts the situation five to six months in advance at the two-month mark, allowing the costs for the three months in between to be eliminated. As a result, capital requirement can also be reduced."**
---
## References 
  https://udn.com/news/story/7239/7555914
  
  https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers
  
