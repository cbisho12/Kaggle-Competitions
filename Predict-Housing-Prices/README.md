# Kaggle-Competitions
These are two of the models that I submitted to Kaggle for the 
kaggle competition to Predict Housing prices that can be found here:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques

--------------------
----- Model #1 ------
----------------------

The categories that I am using are listed in Categories-1.txt. I have one
hot encoded some of the features that are classified and have converted some of them 
to numerical features. The numerical features used are also listed.
RMSLE = 0.13777

--------------------
----- Model #2 ------
----------------------

The categories that I am using are listed in Categories-2.txt. I have one
hot encoded some of the features that are classified and have converted some of them 
to numerical features. The numerical features used are also listed. The major difference
from Model #1 is that more classified features are converted to numerical features and
now I am including some features that are, Tayor expanded by taking the features up to 
powers of 5.
RMSLE = 0.14799



For both of these models I am using pandas dataframes and then converting from
the pandas dataframes to numpy arrays in order to feed the arrays to the regression routine.
For the multiple regression I am using Scikit-learn's Ridge Regression where I have tested what 
is the best coefficient for the L2 Norm. I have included some scratch ipython notebooks where I
am testing the L2 norm as well as k-fold cross validation error. 

