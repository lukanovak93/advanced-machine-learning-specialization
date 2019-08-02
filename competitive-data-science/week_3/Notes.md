# Notes: Week 3

### 1. Evaluation metrics
- Notation:  
    - <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N" title="N" /></a>
 - number of objects
    - <a href="https://www.codecogs.com/eqnedit.php?latex=y&space;\in&space;\R^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;\in&space;\R^N" title="y \in \R^N" /></a> - target values
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;\in&space;\R^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;\in&space;\R^N" title="\hat{y} \in \R^N" /></a> - predictions
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y_i}" title="\hat{y_i}" /></a> - prediction for i-th object
    - <a href="https://www.codecogs.com/eqnedit.php?latex=y_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i" title="y_i" /></a> - target for i-th object

##### 1. Regression metrics
- Mean Squared Error (MSE):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;MSE&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{N}(y_i&space;-&space;\hat{y_i})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;MSE&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{N}(y_i&space;-&space;\hat{y_i})^2" title="MSE = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y_i})^2" /></a>
    - best constant target prediction: target mean
- Root Mean Squared Error (RMSE):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\sqrt{MSE}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\sqrt{MSE}" title="\sqrt{MSE}" /></a>
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;R^2&space;=&space;1&space;-&space;\frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}(y_i&space;-&space;\bar{y})^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;R^2&space;=&space;1&space;-&space;\frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}(y_i&space;-&space;\bar{y})^2}" title="R^2 = 1 - \frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}(y_i - \bar{y})^2}" /></a>
- Mean Absolute Error (MAE):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;MAE&space;=&space;\frac{1}{N}\sum_{i=1}^{N}&space;|&space;y_i&space;-&space;\hat{y_i}&space;|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;MAE&space;=&space;\frac{1}{N}\sum_{i=1}^{N}&space;|&space;y_i&space;-&space;\hat{y_i}&space;|" title="MAE = \frac{1}{N}\sum_{i=1}^{N} | y_i - \hat{y_i} |" /></a>
    - less sensitive of outliers then MSE
    - widely used in finance
    - best constant target prediction: target median
    - gradient = 0 when the prediction is perfect (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat{y_i}&space;-&space;y_i&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{y_i}&space;-&space;y_i&space;=&space;0" title="\hat{y_i} - y_i = 0" /></a>)

### 2. Mean encodings


