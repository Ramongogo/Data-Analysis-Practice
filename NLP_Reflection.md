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
