# Notes: Week 3

### 1. Evaluation metrics
- Notation:  
    - $N$ - number of objects
    - $y \in \R^N$ - target values
    - $\hat{y} \in \R^N$ - predictions
    - $\hat{y_i}$ - prediction for i-th object
    - $y_i$ - target for i-th object

##### 1. Regression metrics
- Mean Squared Error (MSE):
    - $MSE = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y_i})^2$
    - best constant target prediction: target mean
- Root Mean Squared Error (RMSE):
    - $\sqrt{MSE}$
    - $R^2 = 1 - \frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}(y_i - \bar{y})^2}$
- Mean Absolute Error (MAE):
    - $MAE = \frac{1}{N}\sum_{i=1}^{N} | y_i - \hat{y_i} |$
    - less sensitive of outliers then MSE
    - widely used in finance
    - best constant target prediction: target median
    - gradient = 0 when the prediction is perfect ($\hat{y_i} - y_i$)

### 2. Mean encodings


