Halley Paulson and Machi Iwata

Elena Machkasova


# Project Background
The purpose of this project was to create a model that can predict the quality score of white wines based on different measurements. We used the white wine dataset found [here](https://archive.ics.uci.edu/ml/datasets/wine+quality). Through two different neural network approaches, regression and classification, we tried to predict reduce the loss and boost the accuracy in many different ways. Our hypotheses was that the regression model would perform better than the classification model because the quality scores were numerical and the regression model was designed to predict a numerical value.


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

We are using the **quality** column as our labels and predictor variable. We did not find any missing data and we had 4,898 elements in the dataset. We created a correlation plot to determine which features were highly correlated to the predictor variable, but did not find anything that stood out. We decided to use all the available features in the data for training and testing, excluding our labels, in the hopes that the more data there was the better it would predict. 

```{r}
white_wine = read.csv('wine_data/winequality-white.csv', sep=';')
library(psych)
corPlot(white_wine)
```
![image1](https://github.com/halleypaulson/CSCI-4555-Final-Project/blob/main/correlation.png)

# Method
In this project, we are using multilayer perceptron for neural network to predict the white wine quality. <!-- I need your help with the "why" part-->
At first, we had to normalize features because our features had different rages, and we used scale() function to do that. Then we built our model based on training set. We randomly split the dataset into 2 sets which are training set and testing set, and the training set contains 80% of full-set and the testing set has 20%.  


# Result

# Challenges


