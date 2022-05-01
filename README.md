Halley Paulson and Machi Iwata  
CSCI 4555  
Elena Machkasova


# Project Background
The purpose of this project was to create a model that can predict the quality score of white wines based on different measurements. We used the white wine dataset found [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). Through two different task approaches, regression and classification, we tried to reduce the loss and boost the accuracy in many different ways. Our hypotheses was that the regression version of our model would perform better than the classification version because the quality scores were numerical and regression was designed to predict a numerical value.


# Data Description

There are 12 features in the dataset to determine whether the quality of white wine is good or not. List pulled from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/).

1. **fixed acidity**: most acids involved with wine or fixed or nonvolatile
2. **volatile acidity**: the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
3. **citric acid**: found in small quantities, citric acid can add ‘freshness’ and flavor to wines
4. **residual sugar**: the amount of sugar remaining after fermentation stops, it’s rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet
5. **chlorides**: the amount of salt in the wine
6. **free sulfur dioxide**: the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine
7. **total sulfur dioxide**: the amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine
8. **density**: the density of water is close to that of water depending on the percent alcohol and sugar content
9. **pH**: describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale
10. **sulfates**: a wine additive which can contribute to sulfur dioxide gas (S02) levels, which acts as an antimicrobial and antioxidant
11. **alcohol**: the percent alcohol content of the wine
12. **quality**: score between 0 and 10.

![image2](https://github.com/halleypaulson/CSCI-4555-Final-Project/blob/main/design_matrix_sample.png)
We are using the **quality** column as our labels and predictor variable. We did not find any missing data and we had 4,898 elements in the dataset. We created a correlation plot to determine which features were highly correlated to the predictor variable, but did not find anything that stood out. We decided to use all the available features in the data for training and testing, excluding our labels, in the hopes that the more data there was the better it would predict. We also scaled all of our features, excluding our predictor variable, to better fit a normal distribution.

```{r}
white_wine = read.csv('wine_data/winequality-white.csv', sep=';')
library(psych)
corPlot(white_wine)
```
![image1](https://github.com/halleypaulson/CSCI-4555-Final-Project/blob/main/correlation.png)

# Method
In this project, we are using a multilayer perceptron, or feed-forward neural network, to allow for deep learning. We chose to use this since it is the most basic form of deep learning models and our task did not require anything more complicated. 

For the regression version, the training data was 80% of the total data frame and the testing data was the rest. This was because we assumed this split was best practice. For the classification version, we used about 62% of the total data frame for the training data and the rest for testing to see if having more testing data would help the evaluation of the model.

In both the regression and classification versions of our model, ReLU was used as the activation function in all the layers except the output layer. In our regression version, the output layer used a linear activation function and in our classification version, the Softmax activation funciton was used. ReLU was used because it is currently the 'go-to' activation function for many models. Linear and Softmax activation functions were used in order to fit the task we wanted; numerical for regression and category for categorical.

In both version of the model, an Adam optimizer was used because it is the best optimizer in most situations. For the regression model, the Mean Squared Error was used for the loss as this measures the mean squared difference between the predicted and actual value. So, the smaller the loss the better our model would be predicting. For the classification version, Categorical Crossentropy was used as the loss function to suit the nature of the task. Instead of just loss, we measured accuracy as well for this version of the model.  

## Process
The classification version was a pretty close copy to the regression version and many of the trials of getting to the final model were done on the regression version.

In the begining, we first started with a basic model with 3 layers, one input, one hidden and one output, and unprocessed data. It was clear that we needed some type of normalization right away. Our loss was not decreasing steadily, but instead jumping around. After normalizing our data, we saw smooth and steady changes to our loss. Once that was solved, we tried adding more hidden layers and increasing the amount of nodes in each layer. Since we had a lot of features, we found it was beneficial to have more nodes in our layers so we stayed at 100 nodes for each layer. The total amount of hidden layers in the regression version ended up being 4, while the classification version ended up having 3 since we didn't see much change by adding more and it took a lot of time to train with more layers.

For our regression version, we tried two different loss functions. At first we tried the Mean Absolute Error, which didn't seem too different from the Mean Squared Error, but after some research we found that it was common for regression models to use the Mean Squared Error more. Our classification version was based off of an example provided by Elena and we left it unchanged due to the high performance of her example model. We never changed the optimizer because the Adam optimizer is highly recomended.

We tried different combinations of epochs and batch sizes for training and found that the more epochs there were, the better our training metrics were at the end. At one point, our batch size was 15 and our epochs were 60. Our regression version was performing really well, with loss as low as 0.12. We hit a road block when our evaluated loss wasn't matching the loss measured at the end of training. Because of this, we had to retry node amounts, layer amounts, epochs, and batch sizes. We even tried adding dropout rates to no success. What helped match our results was regularization in the hidden layers. We used L1 and L2 regularization in the regression version and only L1 in the classification. While this helped us finally see our evaluated loss match the loss measured after training, we sadly saw an increase in our loss in both versions. With our loss stabilizing after a few epochs, we defaulted to a batch size of 32 with 50 epochs. It was still necessary to have a high amount of epochs, but we didn't want to overdo it.

# Results
In the end, our regression version had an average loss of about 0.57 and our classification version had a loss of about 1.1 and accuracy of about 0.55. Most things we tried didn't do much besides decrease the performance. While typically the lower the loss the better the performance, it seemed that both models had similar performance. Some predictions were close and some weren't. The regression version seemed to be much better at predicting a wine quality of 6 and the lower and upper bounds tended to be innacurate. The classification version had a higher loss, but showed similar results with it sometimes predicting accurately and sometimes not. We would like to say that the regression version's loss means its better at predicting the quality, but it would never get exact values. The classification version, while inaccurate, can still predict a 6 versus the regression model predicting a 5.9. So, it's clear that the classification model should be used to predict quality.

# Challenges
We think our data made this challenging. Trying different subsets of features proved fruitless in changing the loss and accuracy of both versions of our model. While normalization helped, many features still had fairly different ranges. In the end, we question if any of these measurements are good indicators of wine quality. It just didn't seem like the data could be used to pinpoint a certain wine quality. We also think the our model was overfitting since the loss stagnated and we had a pretty complicated model in both versions. If we were to make improvements, we would certainly spend more time researching and improving our classification model rather than our regression. There must be different ways of preprocessing that could improve the model.


