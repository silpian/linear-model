# linear-model

We implement LASSO, OLS, and ridge regression using (cyclical) coordinate descent, gradient descent, and closed form equations. The L1 regularization parameter is chosen through leave-one-out cross-validation.

## Resources
* [Regularization Paths for Generalized Linear Models via Coordinate Descent](https://web.stanford.edu/~hastie/Papers/glmnet.pdf)

## Setup/Dependencies
```
pip install -r requirements.txt
```
Dependencies: numpy

## Use
```python
import linear_model
lasso = linear_model.Lasso()
lasso.fit_by_coordinate_descent(y, X)
lasso.predict(new_X)
```

For more examples of use cases and derivations of the update equations, see [Linear Regression Derivations](https://github.com/silpian/linear-model/blob/master/Linear%20Regression%20Derivations.ipynb)
