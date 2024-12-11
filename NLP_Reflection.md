## Results 
---
>**The final selected stacking model composed of ExtraTreesClassifier, RandomForestClassifier, XGBClassifier and LGBMClassifier has a weighted average f1 score of 0.9862, capturing 93% of customers whose credit cards have been inactive for two months and are likely to terminate the bank's credit card contract due to prolonged inactivity ,and estimating not only to save 55% of management cost of inactive credit cards but also reduce capital requirement.**
---
## Process Elaboration
### 1. Data Preprocessing
* Importing data
* Removing irrelevant columns
* Changing target column's value into integar
* Cleaning data - removing punctuation and stopwords 
  * Original sentence
  
  ![Original Sentence](https://github.com/user-attachments/assets/50ff1503-f1be-432e-aafa-09190e891002)

  * Cleaned sentence
    
  ![螢幕擷取畫面 2024-12-11 140328](https://github.com/user-attachments/assets/cdf53090-e89c-4451-bcb0-960047bce868)

### 2. Text Encoding
  * Using Tokenizer to convert sentences into integer sequences.

    ![螢幕擷取畫面 2024-12-11 140350](https://github.com/user-attachments/assets/b44b1ec0-f447-4ad2-b7bf-844afab9d83a)
  
  * Padding 

    ![螢幕擷取畫面 2024-12-11 140409](https://github.com/user-attachments/assets/2e765b86-562b-40cb-8e0d-2c8b64cae0db)

### 3. Model Building
  * Model Settings
    * model = models.Sequential()
    * model.add(layers.Embedding(20000, 32, input_length=20))  # input
    * model.add(layers.LSTM(64, dropout=0.1))  # LSTM 
    * model.add(layers.Dense(3, activation='softmax'))  # Output
    * model.build(input_shape=(None, 128)) # batch size
  ![螢幕擷取畫面 2024-12-11 140146](https://github.com/user-attachments/assets/08b2f180-2428-4ab8-9941-2b64d8852627)

  


