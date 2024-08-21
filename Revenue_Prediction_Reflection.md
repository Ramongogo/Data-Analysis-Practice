# Results 
---
> **The final Lasso model has an acurracy of 0.672868 with three features selected which are 'Number_of_Customers', 'Menu_Price', and 'Marketing_Spend' .**
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
  
  ![螢幕擷取畫面 2024-08-01 231033](https://github.com/user-attachments/assets/eaaf9af7-fa47-47f4-9946-2627f06c29f1)
## Strategy Formulation
---
> Since these three features have linear realtionship with revenue and 'Number_of_Customers', 'Menu_Price', 'Marketing_Spend''s coefficient with revenue are respectively 75.9053, 25.3290, and 28.1770.
> 
> Also these three features has no collinarity.  
> So I developed one strategy for each feature to try increasing revenue and calculated potential growth. 
---
### 1. Number_of_Customers - Limit Dining Time
Encourage customers to make reservations in advance to avoid long wait times and set a dining time limit, usually between 90 minutes to 2 hours to increase table turnover rate.
### 2. Menu_Price - Pricing Stratification:
Design a diversified menu with different priced set, offering both premium and budget options to cater to different customer segments to increase overall menu's average price.
### 3. Marketing_Spend - Managing a Brand's Social Media Page
Introduce restaurant products and brand story to ensure new customers can immediately understand restaurants's information and build a connection with existing customers.
### Potential Growth 
A rough estimate : Assuming that after implementing these strategies, the three features respectively increase by 10%, 5%, and 10%, it is estimated that revenue can increase by 11.67468%(10% * 75.9053 + 5% * 25.3290 + 10% * 28.1770).
#### Areas For Improvement 
If the data includes seating capacity, daily number of customers, the average menu prices for both stratified and non-stratified pricing, as well as the types of marketing strategies used, a more accurate analysis and estimate can be made.
### References
https://www.kaggle.com/datasets/mrsimple07/restaurants-revenue-prediction
