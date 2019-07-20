# Week 1 notes

### 1. Recap of main ML algorithms

- **Linear models**: try to separate data points with a plane, into 2 subspaces
    - ex: Logistic regression, Support Vector Machines (SVM)
    - Available in Scikit-Learn or Vowpal Wabbit
- **Tree-based models**: use Decision Trees (DT) like Random Forest and Gradient Boosted Decision Trees (GBDT)
    - Applies a “Divide and Conquer” approach by splitting the data into sub-spaces or boxes based on probabilities of outcome
    - In general, DT models are very powerful for tabular data; but rather weak to capture linear dependencies as it requires a lot of splits.
    - Available in Sickit-Learn, XGBoost, LightGBM
- **kNN**: K-Nearest-Neighbors, looks for nearest data points
    - Close objects are likely to have the same labels.
- **Neural Networks**: often seen as a “black-box”, can be very efficient for Images, Sounds, Text and Sequences
    - Available in TensorFlow, PyTorch, Keras (and a lot of other libraries)

- **No Free Lunch Theorem**: there’s not a single method that outperforms all the others for all the tasks.

### 2. Software and Hardware requirements

#### Hardware
- PC with a recent Nvidia GPU, a CPU with 6-cores and 32gb of RAM
- A fast storage (hard drive) is critical, especially for Computer Vision, so a SSD is a must, a NVMe even better. Otherwise, use cloud services like AWS but beware of the operating costs vs. a dedicated PC.

#### Software
- Linux (Ubuntu with Anaconda) is best, some key libraries aren’t available on Windows
    - Python is today’s favorite as it supports a massive pool of libraries for ML
    - Numpy for linear algebra
    - Pandas for dataframes (like SQL)
    - Scikit-Learn for classic ML algorithms
    - Matplotlib for plotting
    - Jupyter Notebook as an IDE (Integrated Development Environment)
    - XGBoost and LightGBM for gradient-boosted decision trees
    - TensorFlow/Keras and PyTorch for Neural Networks

### 3. Feature preprocessing and generation with respect to models

#### Numeric features
- Feature Preprocessing: Decision-Trees (DT) vs non-DT models
    - *Scaling*
        - DT try to find the best split for a feature, no matter the scale
        - kNN, Linear or NN are very sensitive to scaling differences
        - *MinMaxScale* - scale to [0, 1] (*sklearn.preprocessing.MinMaxScaler*)  
        `X = (X — X.min) / (X.max — X.min)`
        - *StandardScale* - scale to mean=0, std=1 (*sklearn.preprocessing.StandardScaler*)  
        `X = (X — X.mean) / X.std`
    - In general case, for a non-DT model, we apply a chosen transformation to ALL numeric features.
    - *outliers* - we can clip for 1st and 99th percentiles, aka “winsorization” in financial data
    - *rank* - can better option than MinMaxScaler in case of outliers present (and unclipped), good for non-DT (scipy.stats.rankdata)
    
    - :exclamation: must be applied to both Train and Test together
    - *Log transform* (`np.log(1 + x)`) OR *raising to the power < 1* (`np.sqrt(x + 2/3)`):
        - bringing too big values closer together, especially good for NN

- Feature Generation: based on Exploaratory Data Analysis (EDA) and business knowledge.
    - feature generations is powered by business knowledge - we won't get far if we randomly generate some features without understanding underlying logic and akgorithms we want to use
    - examples: 
        - easy: with `squared_meter` and `price` features, we can generate a new feature `price/squared_meter` OR generating fractional part of a value (`1.99€` &rarr; `0.99`, `2.49€` &rarr; `0.49`)
        - advanced: generating time interval by a user typing a message (for spambot detection) - no user will post with exactly 10 seconds intervals

#### Categorical and Ordinal features
- Feature Preprocessing:
    - *Label Encoding* - replaces categories by numbers
        - Good for DT, not so for non-DT
        - Alphabetical (sorted): `[S, C, Q]` &rarr; `[2, 1, 3]` with `sklearn.preprocessing.LabelEncoder`
        - Order of Appearance: `[S, C, Q]` &rarr; `[1, 2, 3]` with `Pandas.factorize`
        - Frequency encoding: `[S, C, Q]` &rarr; `[0.5, 0.3, 0.2]`, better for non-DT as it preserves information about value distribution, but still useful even for DT
    - *One-hot Encoding*
        - `pandas.get_dummies`, `sklearn.preprocessing.OneHotEncoder`, `keras.utils.to_categorical`
        - Great for non-DT, plus it’s scaled (min=0, max=1)
        - :exclamation: if too many unique values in category, then one-hot generates too many columns with lots of zero-values.
            - to save RAM, use sparse matrices and store only non-zero elements (tip: if non-zero values far less than 50% total)

#### Datetime and Coordinates features
- *Date & Time*:
    - *Periodicity* (Day number in Week, Month, Year, Season) - used to capture repetitive patterns
    - *Time since* - drug was taken, or last holidays, or numbers of days left before etc.
        - can be row-independent moment (ex: since 00:00:00 UTC, 1 January 1970) or Row-dependent (since last drug taken, last holidays, numbers of days left before etc.)
    - *Difference between dates* - for Churn prediction, like `Last_purchase_date — Last_call_date = Date_diff`
- *Coordinates*:
    - Distance to nearest Point of interest (POI) - subway, school, hospital, police etc.
    - Clusters based on new features and use distance to cluster’s center coordinates
    - Create aggregate stats, such as *Number of Flats in Area* or *Mean Realty Price in Area*

#### Handling missing values
- types of missing values: 
    - NaN
    - empty string (`''`)
    - `-1` (replacing missing values in [0,1])
    - very large number: `-99999`, `999` etc.
- `fillna` approaches:
    - `-999`, `-1`
    - mean & median
    - `isnull` binary feature can be beneficial
    - reconstruct the missing value if possible (best approach)
- do not fill `NaNs` before feature generation - can pollute the data (ex: `time_since` or frequency/label encoding) and skew the model.
- `XGboost` can handle `NaN`
- treating test_set values not present in train data - frequency encoding in train_set can help as it will look for frequency in test_set as well.

### 4. Feature extraction from text and images

#### Bag of Words (BOW)
- `sklearn.feature_extraction.text.CountVectorizer` - creates 1 column per unique word, and counts its occurence per row (phrase)

- text preprocessing:
    - lowercase: Very->very
    - lemmatization: democracy, democratic, democratization -> democracy (requires good dictionary, corpus )
    - stemming: democracy, democratic, democratization -> democr
    - stopwords: get rid of articles, prepositions and very common words, uses NLTK (Natural Language ToolKit)

- N-grams for sequences of words or characters, can help to use local context
    - `sklearn.feature_extraction.text.CountVectorizer` with Ngram_range and analyzer

- tf-idf for postprocessing (required to scale features for non-DT)
    - `sklearn.feature_extraction.text.TfidfVectorizer`
    - `TF`: Term Frequency (in % per row, sum = 1)
    - `iDF`: Inverse Document Frequency (to boost rare words vs. frequent words)
    - there are a number of different tf-idf methods

#### Word Vectors
 - Word Vectors
    - word2vec converts each word to some vector in a space with hundreds of dimensions, creates embeddings between words often used together in the same context
    - most popular: Google word2vec model (300 dimensions)
    - two different methods - `skipgram` and `CBOW`
    - vec(King) - vec(Queen) = vec(Man) - vec(Woman)
    - Other Word Vectors: Glove, FastText
    - Sentences embeddings: Doc2vec
    - there are pretrained models, like on Wikipedia.
    - note: preprocessing can be applied BEFORE using Word2vec


- Comparing BOW vs. word2vec
    - BOW: very large vectors, meaning of each value in vector is known
    - word2vec: smaller vectors, values in vector rarely interpreted, words with similar meaning often have similar embeddings

#### Extracting features from Images with CNNs
*(covered in details in later lessons)*
    - finetuning or transfer-learning
    - data augmentation

