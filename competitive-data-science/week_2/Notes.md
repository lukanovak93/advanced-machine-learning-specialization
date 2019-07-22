# Notes: Week 2

### 1. Exploratory data analysis (EDA)
- process of familiarizing with the data
- build the intuition about the data
- find insights and generate hypotesizes

#### 1. Visualization
##### Single feature statistics
- **histograms** 
    <img align="right" width="200" height="170" title="Histogram" src="https://github.com/lukanovak93/advanced-machine-learning-specialization/blob/master/competitive-data-science/week_2/resources/Screenshot%20from%202019-07-20%2013-54-43.png"/>
    - split feature range into bins and shows how many points fall into each bin
    - can be missleading - vary number of bins
    - aggregates the data so we cannot see if the values are uniqe or there is a lot of duplicate data
    - doesn't show feature interaction but show feature density
    - `plt.hist(x)`
- **index vs. values plot**
    <img align="right" width="200" height="170" title="Index vs. Value plot" src="https://github.com/lukanovak93/advanced-machine-learning-specialization/blob/master/competitive-data-science/week_2/resources/Screenshot%20from%202019-07-20%2013-54-43.png"/>
    - if there are veritcal lines, that means the data is not properly shufled
    - `plt.plot(x, '.')`
    - colorcode classes: `plt.scatter(range(len(x)), x, c=y)`
##### Feature relations
- **scatter plot**
    <img align="right" width="200" height="170" src=""/>
    - `plt.scatter(x1, x2)`
    - also useful for exploring if data distribution in train and test sets is equal
    - doesn't show feature density, but shows feature interaction (corelation)
    - `pd.scatter_matrix(df)` - if number of features is small enough, plot every feature against all other
    <img align="right" width="200" height="170" src=""/>
- **column correlation**
    - `df.corr()`, for custom correlations: `plt.matshow(...)`
    <img align="right" width="200" height="170" src=""/>
    - feature grouping - run clustering on columns and rearange columns
        - creating new features based on groups
    <img align="right" width="200" height="170" src=""/>
    - `df.mean().plot(style='.')` - another method for grouping features
        - better to sort features: `df.mean().sort_values().plot(style='.')`

#### 2. Feature statistics and DataFrame descriptions
- `df.describe()` - infor about mean, std and percentiles of feature distribution
- `x.mean()`, `x.var()`
- `x.value_counts()` - number occurances of distinct feature values
- `x.isnull()` - find missing values in the data

#### 3. Dataset cleaning and things to check
- duplicated and constant features
    - `df.nunique(axis=1) == 1` where `df` is dataframe **before** splitting to train and test sets
- feature constant on train set but has different value on test set &rarr; **remove** feature completely
-  `df.T.drop_duplicates()` - drop duplicated features
- duplicate categorical features with different class names:
```python
for i in categorical_features:
    df[f] = df[f].factorize()

df.T.drop_duplicates()
```
- check if dataset is shuffled
    - plot target vs. row index &rarr; if data is shuffled, oscilation in the plot
    - good idea is to plot the mean as well

### 2. Validation
#### 1. Validation strategies
- Holdout
    - splitting training data into *train* and *validation* datasets
    - for selection the best model &rarr; after selectiong the best model, put all data back together and train the model again on the whole train dataset
    - number of groups for validation = 1
    - `sklearn.model_selection.ShuffleSplit`
- K-fold
    - *'repeted holdout'*
    - data is splitted in **k** parts and in training we iterate through those **folds** and in each fold we have a different validation set - no fold can be the validation set twice
    - after the process &rarr; average scores over **k** folds
    - guaranteed that every sample will be in the validation dataset
    - `sklearn.model_selection.Kfold`
- Leave-one-out
    - basically K-fold with `k = num_samples` (len(data))
    - `sklearn.model_selection.LeaveOneOut`
    - used when there is not enough data and the model is retrained fast enough
- Stratification
    - ensuring similar target distribution over all folds
    - useful for:
        - small datasets
        - unbalanced datasets &rarr; binnary classification if target avg is very close to 0 or 1
        - multiclass classification
#### 2. Data splitting strategies
- logic of feature generation depends on the data splitting strategy
- split the data to mimic train-test split
- Random (rowwise)
    - if rows are independent of eachother
- Timewise
    - sort by date and then take the last part
    - moving window validation
- ID based split
- Combined
    - used multiple features for splitting
    - for example, if we want to predict future sales for a number of shops over the time, we could group the data by each date for a specific shop

#### Practical tips
- if the feature should be of type integer, but when loaded in pandas DataFrame are showed as float &rarr; it means that feature has some `NaN` values (some of the rows have `NaN` in that column)
- if some `ID` column is shared between train and test, it can sometimes be succesfully used to improve the score
- pay attention to columns with `NaNs` and the number of `NaNs` for each row can serve as a nice feature later
    - number of `NaNs` for each row: `train.isnull().sum(axis=1)`
- **competitions** - if the model scores better on train sets then on the validation/test set &rarr; model is **overfitted**
    - this does not apply in the real word problems

- **Validation stage problems**
    - causes of different scores and optimal parametes:
        - too little data
        - too diverse and inconsistent data
    - **solution** - extensive validation:
            - average scores from different KFold splits
            - tune model on one set of splits, evaluate score on other
- **Submission stage problems**
    - leaderboard score is consistently higher/lower then validation score
    - leaderboard score is not corelated with validation score at all
    - reasons
        - too little data in public leaderboard
        - train and test data are from different distributions
    - example from the course - avg man and woman height, train consists only of woman, test only of men - problem is we will predict woman height for men and there is no way to adjust the predictions
        - mean for train - calculate from train data
        - mean for test - leaderboard probing 
        - **solution** - calculate mean for men by leaderboard probing and shift the prediction of the model by `mean_man - mean_woman`
    - more realistic situation
        - train: 90% woman, 10% man
        - test: 10% woman, 90% man
        - **solution** - force the validation to be of the same distribution like the test &rarr; 10% woman, 90% man
    - leaderboard shuffle:
        - randomness
        - small amount of data
        - different public/private distribution - often in time-series data
        - **solution** - trust the validation

### 3. Data leakages
- an unexpected information in the data that allows us to make unrealistically good predictions, you may have think of it as of directly or indirectly adding ground truths into the test data
#### 1. Time series
- split should be done on time
    - in real life, there is no information from future
    - in competition, **first** check if train/public/private split is based on time &rarr; if even one of them is not based on time, there's a data leak
    - even if split by time, features may contain information about future:
        - user history in click-through rate (CTR) tasks
        - weather
- most often- meta data, information in IDs, row order
#### Leaderboard probing
