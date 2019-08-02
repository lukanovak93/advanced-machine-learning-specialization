# Notes: Week 3

### 1. Evaluation metrics

##### 1. Regression metrics

- Notation:  
    - <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N" title="N" /></a>
 - number of objects
    - <a href="https://www.codecogs.com/eqnedit.php?latex=y&space;\in&space;\R^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;\in&space;\R^N" title="y \in \R^N" /></a> - target values (ground truth)
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;\in&space;\R^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;\in&space;\R^N" title="\hat{y} \in \R^N" /></a> - predictions
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y_i}" title="\hat{y_i}" /></a> - prediction for i-th object
    - <a href="https://www.codecogs.com/eqnedit.php?latex=y_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i" title="y_i" /></a> - target for i-th object

- Mean Squared Error (MSE):  

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;MSE&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{N}(y_i&space;-&space;\hat{y_i})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;MSE&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{N}(y_i&space;-&space;\hat{y_i})^2" title="MSE = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y_i})^2" /></a>
    - U-shaped
    - best constant target prediction: target mean
- Root Mean Squared Error (RMSE):  

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\sqrt{MSE}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\sqrt{MSE}" title="\sqrt{MSE}" /></a>
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;R^2&space;=&space;1&space;-&space;\frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}(y_i&space;-&space;\bar{y})^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;R^2&space;=&space;1&space;-&space;\frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}(y_i&space;-&space;\bar{y})^2}" title="R^2 = 1 - \frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}(y_i - \bar{y})^2}" /></a>
- Mean Absolute Error (MAE):  

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;MAE&space;=&space;\frac{1}{N}\sum_{i=1}^{N}&space;|&space;y_i&space;-&space;\hat{y_i}&space;|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;MAE&space;=&space;\frac{1}{N}\sum_{i=1}^{N}&space;|&space;y_i&space;-&space;\hat{y_i}&space;|" title="MAE = \frac{1}{N}\sum_{i=1}^{N} | y_i - \hat{y_i} |" /></a>
    - V-shaped
    - less sensitive of outliers then MSE
    - widely used in finance
    - best constant target prediction: target median
    - gradient = 0 when the prediction is perfect (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat{y_i}&space;-&space;y_i&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{y_i}&space;-&space;y_i&space;=&space;0" title="\hat{y_i} - y_i = 0" /></a>)

- Mean Squared Percentage Error (MSPE):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;MSPE&space;=&space;\frac{100\%}{N}\sum_{i=1}^{N}(\frac{y_i&space;-&space;\hat{y_i}}{y_i})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;MSPE&space;=&space;\frac{100\%}{N}\sum_{i=1}^{N}(\frac{y_i&space;-&space;\hat{y_i}}{y_i})^2" title="MSPE = \frac{100\%}{N}\sum_{i=1}^{N}(\frac{y_i - \hat{y_i}}{y_i})^2" /></a>
    - weighted version of MAE
    - weight of each sample is inversely proportional to target squared
    - best constant target prediction: weighted target mean &rarr; biased to small targets because the absolute error for them is weighted with the highest weight and thus impacts the metric the most
   
- Mean Absolute Percentage Error (MAPE):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;MAPE&space;=&space;\frac{100\%}{N}\sum_{i=1}^{N}\left&space;|\frac{y_i&space;-&space;\hat{y_i}}{y_i}\right&space;|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;MAPE&space;=&space;\frac{100\%}{N}\sum_{i=1}^{N}\left&space;|\frac{y_i&space;-&space;\hat{y_i}}{y_i}\right&space;|" title="MAPE = \frac{100\%}{N}\sum_{i=1}^{N}\left |\frac{y_i - \hat{y_i}}{y_i}\right |" /></a>
    - weighted version of MAE
    - weight of each sample is inversely proportional to its target
    - best constant target prediction: weighted target median &rarr; biased to small targets because the absolute error for them is weighted with the highest weight and thus impacts the metric the most
        - very biased to outliers that have very small values because those outliers will have the highest weight

- for MAPE and MSPE - the cost we pay for fixed absolute error depends on the target values &rarr; as te target increases, we pay less for the same absolute error

- Root Mean Squared Logarithmic Error (RMSLE):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;RMSLE&space;=&space;\sqrt&space;{\frac{1}{N}\sum_{i=1}^{N}(log(y_i&space;&plus;&space;1)&space;-&space;log(\hat{y_i}&space;&plus;&space;1))^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;RMSLE&space;=&space;\sqrt&space;{\frac{1}{N}\sum_{i=1}^{N}(log(y_i&space;&plus;&space;1)&space;-&space;log(\hat{y_i}&space;&plus;&space;1))^2}" title="RMSLE = \sqrt {\frac{1}{N}\sum_{i=1}^{N}(log(y_i + 1) - log(\hat{y_i} + 1))^2}" /></a>
    - used in same situations as MSPE and MAPE - cares more about the relative error then the absolute error
    - assymetric curve - better to predict more then the same amount less then the target: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;weight(y_i&space;-&space;\hat{y_i}&space;<&space;0)&space;>&space;weight(y_i&space;-&space;\hat{y_i}&space;>&space;0)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;weight(y_i&space;-&space;\hat{y_i}&space;<&space;0)&space;>&space;weight(y_i&space;-&space;\hat{y_i}&space;>&space;0)" title="weight(y_i - \hat{y_i} < 0) > weight(y_i - \hat{y_i} > 0)" /></a>

#### 2. Classification metrics
- Notation:  
    - <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N" title="N" /></a>
 - number of objects
 - <a href="https://www.codecogs.com/eqnedit.php?latex=L" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L" title="L" /></a>
 - number of classes
    - <a href="https://www.codecogs.com/eqnedit.php?latex=y&space;\in&space;\R^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;\in&space;\R^N" title="y \in \R^N" /></a> - target values (ground truth)
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;\in&space;\R^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;\in&space;\R^N" title="\hat{y} \in \R^N" /></a> - predictions
    <a href="https://www.codecogs.com/eqnedit.php?latex=[a=b]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[a=b]" title="[a=b]" /></a> - indicator function
    - **Soft labels (predictions)** - classifier's scores (probabilities vector)
    - **Hard labels (predictions)** - mappings from soft labels to actuall class (argmax + threshold)
    
- Accuracy:
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Accuracy&space;=&space;\frac{1}{N}\sum_{i=1}^{N}[\hat{y_i}=y_i]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Accuracy&space;=&space;\frac{1}{N}\sum_{i=1}^{N}[\hat{y_i}=y_i]" title="Accuracy = \frac{1}{N}\sum_{i=1}^{N}[\hat{y_i}=y_i]" /></a>
    - how frequently the class prediction is correct
    - works with hard prediction-
    - best conctant to predict: the most frequent class
    
- Logarithmic loss (logloss):
    - binnary logloss: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Accuracy&space;=&space;-\frac{1}{N}\sum_{i=1}^{N}(y_i&space;log(\hat{y_i})&space;&plus;&space;(1&space;-&space;y_i)log(1&space;-&space;\hat{y_i}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Accuracy&space;=&space;-\frac{1}{N}\sum_{i=1}^{N}(y_i&space;log(\hat{y_i})&space;&plus;&space;(1&space;-&space;y_i)log(1&space;-&space;\hat{y_i}))" title="Accuracy = -\frac{1}{N}\sum_{i=1}^{N}(y_i log(\hat{y_i}) + (1 - y_i)log(1 - \hat{y_i}))" /></a>
    - multiclass logloss: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Accuracy&space;=&space;-\frac{1}{N}\sum_{i=1}^{N}\sum_{l=1}^{L}y_{il}&space;log(\hat{y_{il}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Accuracy&space;=&space;-\frac{1}{N}\sum_{i=1}^{N}\sum_{l=1}^{L}y_{il}&space;log(\hat{y_{il}})" title="Accuracy = -\frac{1}{N}\sum_{i=1}^{N}\sum_{l=1}^{L}y_{il} log(\hat{y_{il}})" /></a>
    - cares how confident classifier was in its prediction - works with soft scores
    - strongly penalizes completely wrong answers - prefers to have a lot of small errors then one big error

- Area Under Curve (AUC ROC or simply AUROC):
    - ROC stands for *Receiver Operating Characteristics*
    - Recall (Sensitivity) - what proportion of actual positives was identified correctly: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\text{Recall}&space;=&space;\frac{TP}{TP&plus;FN}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\text{Recall}&space;=&space;\frac{TP}{TP&plus;FN}" title="\text{Recall} = \frac{TP}{TP+FN}" /></a>
    - Precision - what proportion of positive identifications was actually correct: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\text{Precision}&space;=&space;\frac{TP}{TP&plus;FP}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\text{Precision}&space;=&space;\frac{TP}{TP&plus;FP}" title="\text{Precision} = \frac{TP}{TP+FP}" /></a>
    - AUC is scale-invariant - it measures how well predictions are ranked, rather than their absolute values
        - scale invariance is not always desireable
    - AUC is classification-threshold-invariant - it measures the quality of the model's predictions irrespective of what classification threshold is chosen
        - also not always desireable - in cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error
        
    


### 2. Mean encodings


