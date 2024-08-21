## Results 
---
>**The final selected stacking model composed of LogisticRegression, SVC, XGBClassifier and GaussianNB has a f1 score of 0.623 and a weighted average f1 score of 0.8175, capturing 45.3% of customers who might be potential defaulters,and estimating to save the amount of same percentage of loan default.**
---
## Motive 
---
>**Although the default rate for loans in Taiwanese banks is currently low, the total amount involved is still substantial. For example, Cathay Bank has a delinquency rate of 0.12%, but the total amount is 2.897 billion NTD. Therefore, I try to develop a model that can predict potential loan defaulters based on the huge dataset from Kaggle.**
---
## Process Elaboration
### 1. Data Processing
* Importing data 
* Checking missing and duplicate data
* Using map,getdummy and labelencoder fucntion to transform categorical data
* Observing the features' correlation with bankruptcy
* Removing outliers
### 2. Feature Engineering
* Feature selection 
  * Selcting absolute value of features' correlation higher than 0.08
  ![Figure_2](https://github.com/user-attachments/assets/53469469-ea8d-4a62-b162-e6543ddc1c26)

* Feature extraction 
  * Using PCA (Explained Variance = 95%) to reduce selected features' dimensionality and eliminate collinearity
  ![Figure_3](https://github.com/user-attachments/assets/48b8db24-1ca0-4d58-bf19-ad110dc7368f)
### 3. Handling Imbalanced Dataset
* There is a slight difference between the variable's values, which is a inbalanced dataset
  ![Figure_4](https://github.com/user-attachments/assets/b9507062-a8ac-468d-a99c-6cc0a2a5cb76)
  
* Using SMOTE to randomly create samples (40% of value zero )
  ![Figure_6](https://github.com/user-attachments/assets/3b387995-2add-4390-94f8-179df51b25ee)
### 4. Model Training
* Using 8 different models, then selecting the best performing model - GaussianNB with f1 score of 0.612.
  
  ![Figure_12](https://github.com/user-attachments/assets/28902c84-f351-47f9-a3ae-62d42bc184dc)
  #### GaussianNB Classification Report
  
  ![Figure_10](https://github.com/user-attachments/assets/3294b97e-625c-45d0-8a5a-d16cc836b11d)

  #### GaussianNB Confusion Matrix
  ![Figure_9](https://github.com/user-attachments/assets/172bd3a9-ec43-4b4b-9399-a85147925f3c)
### 5. Model Stacking ✔️
* Forming a stacking model composed of top four performing model with f1 score of 0.624, slighty better than GaussianNB.
  #### Stacking Model Classification Report
  
  ![Figure_13](https://github.com/user-attachments/assets/ffc7e11f-9af0-4521-adfa-5c141c405c64)

  #### Stacking Model's Confusion Matrix
  ![Figure_11](https://github.com/user-attachments/assets/45663959-ce92-4b86-9562-2661609cbacf)
## Reflection 
---
>**The final selected stocking model has a f1 score of 0.624 and a weighted average f1 score of 0.8175. According to its confusion matrix, we can assume that the model can accurately capture and predict 45.3% of customers who probably become defaulters(3264/3264+3401+534). The model is also estimated to save the amount of same percentage of loan default. Although the model's accuracy is not particularly high, the potential savings in losses are already quite substantial."**
---
## References 
  https://www.kaggle.com/datasets/yasserh/loan-default-dataset/code

  
https://www.fsc.gov.tw/userfiles/file/%E9%99%84%E4%BB%B6%E4%B8%89%EF%BC%9A%E6%9C%AC%E5%9C%8B%E9%8A%80%E8%A1%8C%E8%B3%87%E7%94%A2%E5%93%81%E8%B3%AA%E8%A9%95%E4%BC%B0%E5%88%86%E6%9E%90%E7%B5%B1%E8%A8%88%E8%A1%A8113_06.pdf?fbclid=IwY2xjawEyh7tleHRuA2FlbQIxMAABHZoBLp6TD1l5WWABP8HsWeDBX-Ch1YLu-1v3YTXpZuzcFatA_bwevZBUWA_aem_dzaNWu3cPYPq-pmk1gWT3Q
