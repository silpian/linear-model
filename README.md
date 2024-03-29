# linear-model
We implement LASSO, OLS, ridge, and LAD regression using (cyclical) coordinate descent, gradient descent, closed form equations, gradient boosting, and ensemble learning.

## Resources
* [Regularization Paths for Generalized Linear Models via Coordinate Descent](https://web.stanford.edu/~hastie/Papers/glmnet.pdf)
* [Coordinate Descent Soft-Thresholding Update Operator for Lasso](https://stats.stackexchange.com/questions/123672/coordinate-descent-soft-thresholding-update-operator-for-lasso)
* [Ridge Regression](https://www.statlect.com/fundamentals-of-statistics/ridge-regression)

## Setup/Dependencies
```
pip install -r requirements.txt
```
Dependencies: numpy

## Use
### LASSO
```python
import linear_model
lasso = linear_model.Lasso()
lasso.fit_by_coordinate_descent(y, X)
lasso.predict(new_X)
```

### Gradient Boosted OLS
```python
gbols = linear_model.GradientBoostedOLS()
gbols.fit(y, X)
gbols.predict(new_X)
```

### Gradient Boosted OLS with Bagging
```python
ensemble_gbols = ensemble.LinearEnsemble(linear_model.GradientBoostedOLS)
ensemble_gbols.fit(y, X)
ensemble_gbols.predict(new_X)
```

For more examples of use cases and derivations of the update equations, see [Linear Regression Derivations](https://github.com/silpian/linear-model/blob/master/Linear%20Regression%20Derivations.ipynb)
