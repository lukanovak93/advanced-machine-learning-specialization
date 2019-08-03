# Notes: Week 3

### 1. Evaluation metrics

##### 1. Regression metrics

- Notation:  
    - <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N" title="N" /></a> - number of objects
    - <a href="https://www.codecogs.com/eqnedit.php?latex=y&space;\in&space;\R^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;\in&space;\R^N" title="y \in \R^N" /></a> - target values (ground truth)
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y}&space;\in&space;\R^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y}&space;\in&space;\R^N" title="\hat{y} \in \R^N" /></a> - predictions
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y_i}" title="\hat{y_i}" /></a> - prediction for i-th object
    - <a href="https://www.codecogs.com/eqnedit.php?latex=y_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i" title="y_i" /></a> - target for i-th object

- Mean Squared Error (MSE):  

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;MSE&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{N}(y_i&space;-&space;\hat{y_i})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;MSE&space;=&space;\frac{1}{N}&space;\sum_{i=1}^{N}(y_i&space;-&space;\hat{y_i})^2" title="MSE = \frac{1}{N} \sum_{i=1}^{N}(y_i - \hat{y_i})^2" /></a>
    - U-shaped
    - best constant target prediction: target mean
    - use (these all apply for MSE, RMSE, R-squared):
        - tree-based: `XGBoost`, `LightGBM`, `sklearn.RandomForestRegressor`
        - linear models: `sklearn.<>Regressor`, `sklearn.SGDRegressor`, `Vowpal Wabbit (quantile loss)`
        - neural nets: `PyTorch`, `Keras`, ` Tensorflow`, etc.
    - synonim: **L2 loss**
    
- Root Mean Squared Error (RMSE):  

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\sqrt{MSE}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\sqrt{MSE}" title="\sqrt{MSE}" /></a>
- R-squared:
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;R^2&space;=&space;1&space;-&space;\frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}(y_i&space;-&space;\bar{y})^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;R^2&space;=&space;1&space;-&space;\frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}(y_i&space;-&space;\bar{y})^2}" title="R^2 = 1 - \frac{MSE}{\frac{1}{N}\sum_{i=1}^{N}(y_i - \bar{y})^2}" /></a>

- Mean Absolute Error (MAE):  

    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;MAE&space;=&space;\frac{1}{N}\sum_{i=1}^{N}&space;|&space;y_i&space;-&space;\hat{y_i}&space;|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;MAE&space;=&space;\frac{1}{N}\sum_{i=1}^{N}&space;|&space;y_i&space;-&space;\hat{y_i}&space;|" title="MAE = \frac{1}{N}\sum_{i=1}^{N} | y_i - \hat{y_i} |" /></a>
    - V-shaped
    - less sensitive of outliers then MSE
    - widely used in finance
    - best constant target prediction: target median
    - gradient = 0 when the prediction is perfect (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat{y_i}&space;-&space;y_i&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{y_i}&space;-&space;y_i&space;=&space;0" title="\hat{y_i} - y_i = 0" /></a>)
    - use:
        - tree-based: ~~`XGBoost`~~ (not implemented since second derivative is not defined), `LightGBM`, `sklearn.RandomForestRegressor` - slow
        - linear models: `Vowpal Wabbit (quantile loss)`
        - neural nets: `PyTorch`, `Keras`, ` Tensorflow`, etc.
   - synonims: **L1 loss, Median regression**
   - **Huber loss** - MAE for large errors and MSE for small errors (constant `delta` to determine the magnitude (or threshold) when to start using MAE arround small values)

- Mean Squared Percentage Error (MSPE):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;MSPE&space;=&space;\frac{100\%}{N}\sum_{i=1}^{N}(\frac{y_i&space;-&space;\hat{y_i}}{y_i})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;MSPE&space;=&space;\frac{100\%}{N}\sum_{i=1}^{N}(\frac{y_i&space;-&space;\hat{y_i}}{y_i})^2" title="MSPE = \frac{100\%}{N}\sum_{i=1}^{N}(\frac{y_i - \hat{y_i}}{y_i})^2" /></a>
    - weighted version of MAE
    - weight of each sample is inversely proportional to target squared
    - best constant target prediction: weighted target mean &rarr; biased to small targets because the absolute error for them is weighted with the highest weight and thus impacts the metric the most
    - use:
        - use weights for samples (`sample_weights`) and optimize MSE (XGBoost, LightGBM)
        - another approach would be to sample the train set (`df.sample(weights=sample_weights)`) and feed the model with MSE loss (resample many times and average)
   
- Mean Absolute Percentage Error (MAPE):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;MAPE&space;=&space;\frac{100\%}{N}\sum_{i=1}^{N}\left&space;|\frac{y_i&space;-&space;\hat{y_i}}{y_i}\right&space;|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;MAPE&space;=&space;\frac{100\%}{N}\sum_{i=1}^{N}\left&space;|\frac{y_i&space;-&space;\hat{y_i}}{y_i}\right&space;|" title="MAPE = \frac{100\%}{N}\sum_{i=1}^{N}\left |\frac{y_i - \hat{y_i}}{y_i}\right |" /></a>
    - weighted version of MAE
    - weight of each sample is inversely proportional to its target
    - best constant target prediction: weighted target median &rarr; biased to small targets because the absolute error for them is weighted with the highest weight and thus impacts the metric the most
        - very biased to outliers that have very small values because those outliers will have the highest weight
    - use:
        - use weights for samples (`sample_weights`) and optimize MAE (XGBoost, LightGBM)
        - another approach would be to sample the train set (`df.sample(weights=sample_weights)`) and feed the model with MAE loss (resample many times and average)

- for MAPE and MSPE - the cost we pay for fixed absolute error depends on the target values &rarr; as te target increases, we pay less for the same absolute error

- Root Mean Squared Logarithmic Error (RMSLE):
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;RMSLE&space;=&space;\sqrt&space;{\frac{1}{N}\sum_{i=1}^{N}(log(y_i&space;&plus;&space;1)&space;-&space;log(\hat{y_i}&space;&plus;&space;1))^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;RMSLE&space;=&space;\sqrt&space;{\frac{1}{N}\sum_{i=1}^{N}(log(y_i&space;&plus;&space;1)&space;-&space;log(\hat{y_i}&space;&plus;&space;1))^2}" title="RMSLE = \sqrt {\frac{1}{N}\sum_{i=1}^{N}(log(y_i + 1) - log(\hat{y_i} + 1))^2}" /></a>
    - used in same situations as MSPE and MAPE - cares more about the relative error then the absolute error
    - assymetric curve - better to predict more then the same amount less then the target: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;weight(y_i&space;-&space;\hat{y_i}&space;<&space;0)&space;>&space;weight(y_i&space;-&space;\hat{y_i}&space;>&space;0)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;weight(y_i&space;-&space;\hat{y_i}&space;<&space;0)&space;>&space;weight(y_i&space;-&space;\hat{y_i}&space;>&space;0)" title="weight(y_i - \hat{y_i} < 0) > weight(y_i - \hat{y_i} > 0)" /></a>
    - use:
        - train set - transform target (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;z_i&space;=&space;log(y_i&space;&plus;&space;1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;z_i&space;=&space;log(y_i&space;&plus;&space;1)" title="z_i = log(y_i + 1)" /></a>) and fit the model with MSE loss
        - test set - transform predictions back (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat{y_i}&space;=&space;exp(\hat{z_i})&space;-&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat{y_i}&space;=&space;exp(\hat{z_i})&space;-&space;1" title="\hat{y_i} = exp(\hat{z_i}) - 1" /></a>)

#### 2. Classification metrics
- good explanation [here](http://queirozf.com/entries/evaluation-metrics-for-classification-quick-examples-references)
- Notation:  
    - <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N" title="N" /></a> - number of objects
    - <a href="https://www.codecogs.com/eqnedit.php?latex=L" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L" title="L" /></a> - number of classes
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
    - use:
        - binnary task: use any metric and fit the threshold
        - multiclass task: fit any metric and tune parameters and compare models by accuracy (not metric)
        - zero-one loss, hinge loss (SVM), logistic function (loss, used in logistinc regression)
    
- Logarithmic loss (logloss):
    - binnary logloss: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Accuracy&space;=&space;-\frac{1}{N}\sum_{i=1}^{N}(y_i&space;log(\hat{y_i})&space;&plus;&space;(1&space;-&space;y_i)log(1&space;-&space;\hat{y_i}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Accuracy&space;=&space;-\frac{1}{N}\sum_{i=1}^{N}(y_i&space;log(\hat{y_i})&space;&plus;&space;(1&space;-&space;y_i)log(1&space;-&space;\hat{y_i}))" title="Accuracy = -\frac{1}{N}\sum_{i=1}^{N}(y_i log(\hat{y_i}) + (1 - y_i)log(1 - \hat{y_i}))" /></a>
    - multiclass logloss: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Accuracy&space;=&space;-\frac{1}{N}\sum_{i=1}^{N}\sum_{l=1}^{L}y_{il}&space;log(\hat{y_{il}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Accuracy&space;=&space;-\frac{1}{N}\sum_{i=1}^{N}\sum_{l=1}^{L}y_{il}&space;log(\hat{y_{il}})" title="Accuracy = -\frac{1}{N}\sum_{i=1}^{N}\sum_{l=1}^{L}y_{il} log(\hat{y_{il}})" /></a>
    - cares how confident classifier was in its prediction - works with soft scores
    - strongly penalizes completely wrong answers - prefers to have a lot of small errors then one big error
    - use:
        - tree-based: `XGBoost`, `LightGBM` - calibrate predictions (Platt scaling = fit Logistic regression on prediction, Isotonic regression = fit Isotonic regression on predictions, Stacking)
        - linear models: `sklearn.<>Regressor`, `skleard.SGDRegressor`, `Vowpal Wabbit`
        - neural nets: `PyTorch`, `Keras`, `Tensorflow`

- Area Under Curve (AUC ROC or simply AUROC):
    - ROC stands for *Receiver Operating Characteristics*
    - Recall (Sensitivity) - what proportion of actual positives was identified correctly: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\text{Recall}&space;=&space;\frac{TP}{TP&plus;FN}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\text{Recall}&space;=&space;\frac{TP}{TP&plus;FN}" title="\text{Recall} = \frac{TP}{TP+FN}" /></a>
    - Precision - what proportion of positive identifications was actually correct: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\text{Precision}&space;=&space;\frac{TP}{TP&plus;FP}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\text{Precision}&space;=&space;\frac{TP}{TP&plus;FP}" title="\text{Precision} = \frac{TP}{TP+FP}" /></a>
    - AUC is scale-invariant - it measures how well predictions are ranked, rather than their absolute values
        - scale invariance is not always desireable
    - AUC is classification-threshold-invariant - it measures the quality of the model's predictions irrespective of what classification threshold is chosen
        - also not always desireable - in cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error
    - depends on the order of the targets
    - use:
        - pairwise loss: logloss - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;loss&space;=&space;-\frac{1}{N_0N_1}\sum_{i:y_i=1}^{N_1}\sum_{j:y_j=1}^{N_0}log(prob(\hat{y_j}&space;-&space;\hat{y_i}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;loss&space;=&space;-\frac{1}{N_0N_1}\sum_{i:y_i=1}^{N_1}\sum_{j:y_j=1}^{N_0}log(prob(\hat{y_j}&space;-&space;\hat{y_i}))" title="loss = -\frac{1}{N_0N_1}\sum_{i:y_i=1}^{N_1}\sum_{j:y_j=1}^{N_0}log(prob(\hat{y_j} - \hat{y_i}))" /></a>
        - `XGBoost`, `LightGBM`, neural nets (not out of the box but easy to implement)
    
- Weighted Cohen's Kappa:
    - not so well explained in the course materials but the good explanation can be found [here](https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english)
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;p_e&space;=&space;\frac{1}{N^2}\sum_{k}n_{k1}n_{k2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;p_e&space;=&space;\frac{1}{N^2}\sum_{k}n_{k1}n_{k2}" title="p_e = \frac{1}{N^2}\sum_{k}n_{k1}n_{k2}" /></a>
        - factors in the sum are marginal sums for each of the classes k (one factor is the ground truth labes and other one is classes predicted by the classifier)
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Cohen's&space;Kappa&space;=&space;1&space;-&space;\frac{1-accuracy}{1-p_e}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Cohen's&space;Kappa&space;=&space;1&space;-&space;\frac{1-accuracy}{1-p_e}" title="Cohen's Kappa = 1 - \frac{1-accuracy}{1-p_e}" /></a>
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Weighted&space;Kappa&space;=&space;1&space;-&space;\frac{weighted&space;error}{weighted&space;baseline&space;error}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Weighted&space;Kappa&space;=&space;1&space;-&space;\frac{weighted&space;error}{weighted&space;baseline&space;error}" title="Weighted Kappa = 1 - \frac{weighted error}{weighted baseline error}" /></a>
    - use:
        - optimize MSE and find right thresholds:
            - `1 - MSE`
            - finding the right threshold (not just np.round(predictions) but do grid search - optimize thresholds)
    
#### 3. Metric vs. Loss
- target metric - function that we want to optimize and the that is used to evaluate the model's quality
- optimization loss - what model actually optimizes and uses to asses its quality
- synonims: lost, cost, objective
- some metrics can be optimized directly - MSE, logloss
- some metrics cannot be directly optimized - preproces train set and optimize another metric - MSPE, MAPE, RMSLE
- sometimes, it can be beneficial to optimize another metric and then postprocess the predictions - Accuracy, Kappa
- sometimes there is a need to write own (custom) loss function
- in some cases, another metric can be optimized and used with early stopping
    - early stopping in this case means:
        - set the model to optimize any loss function it can optimize
        - monitor the performance (quality of predictions) on the desired metric
        - stop the training when model starts to overfit the **desired metric** (not the one that the mdel itself is actually optimizing)

### 2. Mean encodings
- [CatBoost](https://github.com/catboost/catboost) library - Gradient boosting on decision trees (supports GPU as well as CPU computations)
- using target to generate features
- ways of using the target variable to compute useful features:
    - notation:
        - `Goods` - number of `1`s (in target column) in the group
        - `Bads` - number of `0`s (in target column) in the group
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;likelihood&space;=&space;\frac{Goods}{Goods&space;&plus;&space;Bads}&space;=&space;mean(target)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;likelihood&space;=&space;\frac{Goods}{Goods&space;&plus;&space;Bads}&space;=&space;mean(target)" title="likelihood = \frac{Goods}{Goods + Bads} = mean(target)" /></a>
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;weight&space;of&space;Evidence=&space;ln(\frac{Goods}{Bads})&space;*&space;100" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;weight&space;of&space;Evidence=&space;ln(\frac{Goods}{Bads})&space;*&space;100" title="weight of Evidence= ln(\frac{Goods}{Bads}) * 100" /></a>
    - sum of `Goods` in the target column for a group
    - diff - `Goods - Bads`
- advantages:
    - compact transformations of categorical variables
    - powerful basis for feature engineering
- disadvantages:
    - need careful validation as there is a lot of ways to overfit
    - significant improvements only on specific datasets

#### Regularization
- CV loop:
    - robust and intuitive
    - 4 or 5 folds are usually enough to get decent results
    - careful with extreme situations like leave one out
- Smoothing:
    - if a group has a lot of samples - we can trust estimated encoding, and reverse otherwise
    - <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\frac{mean(target)*&space;n_{rows}\&space;&plus;\&space;globalmean*\alpha}{n_{rows}\&space;&plus;\&space;\alpha}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{mean(target)*&space;n_{rows}\&space;&plus;\&space;globalmean*\alpha}{n_{rows}\&space;&plus;\&space;\alpha}" title="\frac{mean(target)* n_{rows}\ +\ globalmean*\alpha}{n_{rows}\ +\ \alpha}" /></a>
    - alpha controls the amount of regularization
    - only works with some other regularization method
- Adding noise:
    - degrades the quality of encoding
    - unstable - how much noise? (too much &rarr; feature unusable, too little &rarr; worse regularization)
    - usually used with leave one out (LOO) - neads some hyperparameters tuning &rarr; do not use if there is no time
- Expanding mean
    - idea:
        - fix some sorting of the data
        - use only the rows from `0` to `n-1` to calculate encoding for row `n`
    - least amount of leakage
    - no hyperparameters
    - irregular encoding quality &rarr; bad
- Generalizations and practical examples
    - for pratical use: CV loops and expanding mean
    - regression - median, percentiles, std, distribution bins (regularize all these features)
    - milticlass classification - introducing new information
    - many-to-many relations - statistics on vectors (users-apps example)
    - timeseries - rolling statistics of target variable
    - **Correct validation**:
        - local experiments:
            - estimate encodings  on X_train
            - map encodings to X_train and X_validation
            - regularize on X_train
            - validate model on X_train/X_validation split
        - submission:
            - estimate encodings on all train data
            - map them to train and test
            - regularize on train set
            - fit on train set
