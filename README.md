Halley Paulson and Machi Iwata  
CSCI 4555  
Elena Machkasova


# Project Background
The purpose of this project was to create a model that can predict the quality score of white wines based on different measurements. We used the white wine dataset found [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). Through two different task approaches, regression and classification, we tried to reduce the loss and boost the accuracy in many different ways. Our hypotheses was that the regression version of our model would perform better than the classification version because the quality scores were numerical and regression was designed to predict a numerical value. To be honest, we really just wanted to see if regression could be as good as classification in a situation where the categories were numeric.

# Data Description

There are 12 features in the dataset to determine whether the quality of white wine is good or not. List pulled from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/).

![image2](https://github.com/halleypaulson/CSCI-4555-Final-Project/blob/main/design_matrix_sample.png)

- **fixed acidity**: most acids involved with wine or fixed or nonvolatile
- **volatile acidity**: the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
- **citric acid**: found in small quantities, citric acid can add ‘freshness’ and flavor to wines
- **residual sugar**: the amount of sugar remaining after fermentation stops, it’s rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet
- **chlorides**: the amount of salt in the wine
- **free sulfur dioxide**: the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine
- **total sulfur dioxide**: the amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine
- **density**: the density of water is close to that of water depending on the percent alcohol and sugar content
- **pH**: describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale
- **sulfates**: a wine additive which can contribute to sulfur dioxide gas (S02) levels, which acts as an antimicrobial and antioxidant
- **alcohol**: the percent alcohol content of the wine
- **quality**: score between 0 and 10.

We are using the **quality** column as our labels and predictor variable. We did not find any missing data and we had 4,898 elements in the dataset. We created a correlation plot to determine which features were highly correlated to the predictor variable, but did not find anything that stood out. We decided to use all the available features in the data for training and testing, excluding our labels, in the hopes that the more data there was the better it would predict. We also scaled all of our features, excluding our predictor variable, to better fit a normal distribution and allow for faster convergence.

![image1](https://github.com/halleypaulson/CSCI-4555-Final-Project/blob/main/correlation.png)

# Regression
## Model
```{r}
model = keras_model_sequential() %>%
  layer_dense(units = 100, activation = "relu", input_shape = ncol(reg_training_data)) %>%
  layer_dense(units = 100, activation = "relu", kernel_regularizer=keras$regularizers$L1L2(0.01,0.01)) %>%
  layer_dense(units = 100, activation = "relu",kernel_regularizer=keras$regularizers$L1L2(0.01,0.01)) %>%
  layer_dense(units = 100, activation = "relu") %>%
  layer_dense(units = ncol(reg_training_labels), activation = "linear")

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
)

model %>% 
fit(
  x = reg_training_data,
  y = reg_training_labels,
  epochs = 45,
)
```
For the most part, we chose activation and loss functions based on what was common practice. Since ReLU is very popular and allows for backpropogation, we used it. Since a linear activation function is commonly used in regression so it doesn't alter the outputted value, we used it. After many experiments, we found this model to work the best. We trained using 80% of the dataset and tested with the other 20% because a lot of other models we've seen have used this ratio and it worked well for them.

## Process
What we tried:
- Different epochs: [15,20,30,45,50,60]
- Differenct batch sizes: [15,40,50,60]
- Different amounts of nodes
- Different amounts of layers: 3 vs 4 vs 5 hidden
- MSE vs MAE loss functions
- Regularization: L1, L2 and L1L2
- Dropout

What we found out:
- The more epochs, the better the metrics after training. However, weights can run away and cause misleading results like when our evaluated loss was much worse than the loss measured after training. 
- Smaller batch sizes in combination with a lot of epochs similarly improved metrics but caused misleading results.
- More nodes improved performance over less but cause a risk of overfitting.
- 4 hidden layers is that sweet spot between being too much and not being enough.
- MSE loss is the measurement of the difference between the predicted and the actual value, and is the prefered loss function for regression.
- Regularization fixed the difference in after training loss to evaluated loss and gave very stable training results. While there wasn't a huge difference between L1, L2 and L1L2, we decided to go with L1L2 to be safe.
- Dropout decreased performance.


# Classification
## Model
```{r}
model = keras_model_sequential() %>%
  layer_dense(units = 50, activation = "relu", input_shape = ncol(clas_training_data)) %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dense(units = 50, activation = "relu") %>%
  layer_dense(units = ncol(clas_training_labels), activation = "softmax")

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

model %>% 
fit(
  x = clas_training_data,
  y = clas_training_labels,
  epochs = 45,
)
```
Similarly to the regression version, we used activation and loss functions that are commonly used by other successful models. After many trials and errors, we found that 100 nodes in each layer actually hurt our performance. We also found that dropout increased performance more than regularization. We also trained with 80% of the dataset and then tested with the remaining 20%.

## Process
What we tried:
- Different dropout %: [0.1,0.2,0.3]
- Different amounts of nodes
- Different amounts of layers
- Regularization: L1, L2, and L1L2
- Different epochs
- Different batch sizes
- Different ratios of training and testing data

What we found:
- Our model was likely overfitting. Dropout helped more than regularization and actually decreased loss.
- Less nodes but more layers helped correct overfitting and improved accuracy.
- Similar to regularization, we had an issue with the after training metrics not matching the evaluation metrics. Too many epochs and too small of batch sizes tended to make this issue worse. While regularization fixed it entirely, we removed it for dropout. So, we had to be careful about having too many epochs.
- It was better to have the 80/20 ratio for training and testing data vs 60/40.

# Results
In the end, our regression version had an average loss of about 0.57 and our classification version had a loss of about 0.9 and accuracy of about 0.6. While typically the lower the loss the better the performance, it seemed that both models had similar performance. Some predictions were close and some weren't. The regression version seemed to be much better at predicting a wine quality of 6 and the lower and upper bounds tended to be innacurate. The classification version had a higher loss, but showed similar results with it sometimes predicting accurately and sometimes not. The regression version had much more training stability. We would like to say that the regression version's loss means its better at predicting the quality, but it would never get exact values. The classification version, while inaccurate, can still predict a 6 versus the regression model predicting a 5.9. So, it's clear that the classification model should be used to predict quality. Even though our hypothesis was wrong, we are satisfied with the performance of the regression model.

# Challenges
We think our data made this challenging. Trying different subsets of features proved fruitless in changing the loss and accuracy of both versions of our model. While normalization and dropout helped, it couldn't change the correlation results. In the end, we question if any of these measurements are good indicators of wine quality. It just didn't seem like the data could be used to pinpoint a certain quality. It's difficult because the features were all measurements but the quality was a person's rating based on a taste test. It could be that different combinations of features makes a quality of 6. It could also be that different people rate differently which would certainly mess with predictions. There's also a potential that the models were overfitting, especially since the regression's loss stagnated. If we were to make improvements, we would certainly spend more time researching and improving our classification model rather than our regression. We think there are different ways of preprocessing that could improve the model, but we may also find that it's just bad data.


