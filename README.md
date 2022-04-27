Halley Paulson and Machi Iwata

Elena Machkasova


# Project Background
The purpose of this study is to generate insights into each of the factors in our model's white wine quality and to determine which features of white (red) wines are establishing the quality indicators. According to this website, there are over 10,000 wine varieties in the world, even considering only the grapes kinds, therefore, for many people, figuring out which wine is a good quality wine can be difficult. By using a neural network, we can build regression model and classification model to predict whether a particular wine is good quality or not. For this project, we used the data from the website of the UCI Machine Learning Repository. (https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/). Our hypotheses for this topic are that 


# Data Description

There are 12 features in the dataset to determine whether the quality of white wine is good or not. Here is the list:

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

We did not find any missing data and we have 58,776 elements in the dataset. We created a correlation plot to determine which features are highly correlated to the variable "quality". With this correlation plot, we can see which features are correlated to the quality. Even the variable which has the highest correlation with quality does not show the strong correlation which the number is 0.44 and the variable is alcohol, so we decided to use all the features for this project.

<!-- alcohol, volatile.acidity. sulphates total.sulfur.dioxide might be the top features we want to have-->

```{r}
white_wine = read.csv('wine_data/winequality-white.csv', sep=';')
library(psych)
corPlot(white_wine)
```

# Method
In this project, we are using multilayer perceptron for neural network to predict the white wine quality. <!-- I need your help with the "why" part-->
At first, we had to normalize features because our features had different rages, and we used scale() function to do that. Then we built our model based on training set (80% of full-set). We randomly split the dataset into 2 sets which are training set and testing set, and the training set contains 80% of full-set and the testing set has 20%.  





# Result





