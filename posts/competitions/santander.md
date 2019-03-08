<p align="center"><a href="https://www.kaggle.com/santander/competitions"><img width="50%" src="https://graffica.info/wp-content/uploads/2018/03/santander_logo.png" /></a></p>


Year | Competition | Problem | Evaluation
-----|-------------|---------|:-----------:
2016 | [**Customer Satisfaction**](https://www.kaggle.com/c/santander-customer-satisfaction) </br> Which customers are happy customers? | **Binary classification**  </br> (0=happy, 1=unhappy) | AUC
2017 | [**Product Recommendation**](https://www.kaggle.com/c/santander-product-recommendation) </br> Can you pair products with people? | **Recommendation System** </br> Return a ranked ordering of items| [MAP@7](http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html)
2018 | [**Value Prediction**](https://www.kaggle.com/c/santander-value-prediction-challenge) </br> Predict the value of transactions for potential customers. | **Regression** </br> Amount of money transacted | RMSLE
2019 | [**Customer Transaction Prediction**](https://www.kaggle.com/c/santander-customer-transaction-prediction) </br> Can you identify who will make a transaction? | **Binary classification** </br> (0=won't make, 1=will make) | AUC


## 2016 Customer Satisfaction [3rd place solution](https://www.kaggle.com/c/santander-customer-satisfaction/discussion/20978) 😊😡
- **Goal**: Predict if the customers are satisfied or dissatisfied.
- The solution mostly based on the ensembling of different models
#### Preprocessing
- replacing some values by NA
- dropping sparse and duplicated features
- normalization
#### Feature engineering
- Sum of zeros feature
- t-SNE features
- PCA features
- K-means features
- Likelihood features
#### Models
- Follow The (Proximally) Regularized Leader
- Regularized Greedy Forest
- Random Forests
- ExtraTreesClassifier
- Adaboost trees
- XGBOOST
- neural networks
- KNN


## 2017 [Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation) 📦👍

## 2018 [Value Prediction](https://www.kaggle.com/c/santander-value-prediction-challenge) 💰

## 2019 [Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction) 💳
