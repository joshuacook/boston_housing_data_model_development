# Housing Model Development

This repository began as a Udacity Nanodegree project working on the Boston Housing Data. Final work for this early work in machine learning can be viewed [here](doc/boston_housing.pdf). 

Later, this was reworked with a deeper emphasis on linear models. 

## Boston Redux

First, a [baseline](ipynb/boston_redux-00-baseline.ipynb) model was assessed against three naive models:

 - linear regression with no regularization (ordinary least squares)
 - linear regresison with $\ell_1$ regularization (LASSO)
 - linear regression with $\ell_2$ regularization (Ridge Regression)
 
The results of this were


| alpha   | model                 | test_score | train_score  |
|:-------:|:---------------------:|:----------:|:------------:|
| NaN     | linearregression      | 0.711009   | 0.743956     |
| 0.00001 | lasso                 | 0.711009   | 0.743956     |
| 0.01000 | ridge                 | 0.711016   | 0.743956     |
