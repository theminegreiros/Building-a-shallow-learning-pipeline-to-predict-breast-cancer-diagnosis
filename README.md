# Building a shallow learning pipeline to predict breast cancer diagnosis

This pipeline tutorial was published on medium, you can check it [here](https://theminegreiros.medium.com/building-a-shallow-learning-pipeline-to-predict-breast-cancer-diagnosis-838f88dee327). 

* Full Pipeline code:
  * breast_cancer_pipeline.ipynb
* Breast Cancer Dataset:
  * data_breast_cancer.csv
  
 Cancer is the second leading cause of death globally, being the second major cause of women’s death. More than 9.8 million people died due to cancer only in 2018. Cancer has also a significant and increasing economic impact. The total annual economic cost of cancer was estimated at approximately 1.16 trillion USD in 2010. In the last few decades, several machine learning techniques have been developed for breast cancer detection and classification.
 
 In this article you will learn how to build a machine learning pipeline to classify the Wisconsin Breast Cancer dataset using python.
 
 Don’t worry if you are a data science/machine learning rookie. This article will cover the basic steps to build a complete pipeline to test different machine learning models and get the best one.

We will use the Wisconsin Breast Cancer dataset available at: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data.

We strongly recommend Google Colab or JupyterLab to write your code. If you are going to use JupyterLab, be sure to install all required libraries before running the following code snippets. The full pipeline code is available as a python notebook here.

## **Import python Libraries**

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

```

## Load, check and prepare the data

Let’s load and check the first rows from the dataset.

```python
data = pd.read_csv("data_breast_cancer.csv")
print(data.shape, end = "\n\n\n")
print(data.head())

```

The dataset has 569 rows and 32 columns. The first column seems not to be informative, and the classes we want to predict are in the second column.

The pd.info() function is very important. You should always use it to check for useful information, such as column names and data types, as well as the number of non-null values in each column.

```python

print(data.info())
output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 569 entries, 0 to 568
Data columns (total 32 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   id                       569 non-null    int64  
 1   diagnosis                569 non-null    object 
 2   radius_mean              569 non-null    float64
 3   texture_mean             569 non-null    float64
 4   perimeter_mean           569 non-null    float64
 5   area_mean                569 non-null    float64
 6   smoothness_mean          569 non-null    float64
 7   compactness_mean         569 non-null    float64
 8   concavity_mean           569 non-null    float64
 9   concave points_mean      569 non-null    float64
 10  symmetry_mean            569 non-null    float64
 11  fractal_dimension_mean   569 non-null    float64
 12  radius_se                569 non-null    float64
 13  texture_se               569 non-null    float64
 14  perimeter_se             569 non-null    float64
 15  area_se                  569 non-null    float64
 16  smoothness_se            569 non-null    float64
 17  compactness_se           569 non-null    float64
 18  concavity_se             569 non-null    float64
 19  concave points_se        569 non-null    float64
 20  symmetry_se              569 non-null    float64
 21  fractal_dimension_se     569 non-null    float64
 22  radius_worst             569 non-null    float64
 23  texture_worst            569 non-null    float64
 24  perimeter_worst          569 non-null    float64
 25  area_worst               569 non-null    float64
 26  smoothness_worst         569 non-null    float64
 27  compactness_worst        569 non-null    float64
 28  concavity_worst          569 non-null    float64
 29  concave points_worst     569 non-null    float64
 30  symmetry_worst           569 non-null    float64
 31  fractal_dimension_worst  569 non-null    float64
dtypes: float64(30), int64(1), object(1)
memory usage: 142.4+ KB
None

```

Except for the diagnosis column, the other columns are numerical, depicting mostly float values. This was expected as the dataset description states that the features were computed from digitized images. As machine learning models deal only with numerical data as input, the diagnosis column will be encoded later.

There is no missing data in this breast cancer dataset. However, missing data is very common in medical data. One simple approach is to drop the lines with missing data. But this is not a good way to overcome it, as dropping lines can make your model lose information, consequently affecting the generalization performance. One better solution is to impute missing data. This is a very common and useful approach to deal with missing data in medical datasets. There are good options in sklearn such as [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) and [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html).

## Analyzing the target column:
Let’s explore the target column.

```python

print(data.diagnosis.value_counts(), end = '\n\n\n')
output:
B    357
M    212
Name: diagnosis, dtype: int64

```
As you can see, The Benign class has more samples. Imbalanced class can be a problem. You can check [here](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/) how to deal with imbalanced classes.

## Checking duplicated data:
Duplicated rows must be removed. Duplicates may negatively affect your fitted model.

```python

print("The dataset contain {} duplicates".format(sum(data.duplicated())))
output:
The dataset contain 0 duplicates

```

## Detecting and removing outliers
Removing outliers can improve machine learning performance.

```python

data.drop(['id'], axis = 1, inplace = True)
# numerical data
x = data.select_dtypes("float64")
# identify outliers in the dataset
lof = LocalOutlierFactor()
outlier = lof.fit_predict(x)
mask = outlier != -1
print("Original shape: {}".format(data.shape), end = '\n\n\n')
print("Shape after outliers removal: {}".format(data.loc[mask,:].shape), end = '\n\n\n')
if (data.shape[0] > data.loc[mask,:].shape[0]):
  print("The numerical columns contain outliers")
else:
  print("The numerical columns DO NOT contain outliers")
data = data.loc[mask,:]

```
## Encoding the target column

```python

# define a categorical encoding for target variable
le = LabelEncoder()
# fit and transoform y_train
data["diagnosis"] = le.fit_transform(data.diagnosis)
data.head()

```
## Checking features correlation

```python

correlations = data.corr()
plt.figure(figsize = (30, 20))
sns.heatmap(data = correlations, annot = True)
output:

```

![heatmap_medium](https://user-images.githubusercontent.com/40517814/187779307-5e73d013-6119-44d1-8cdc-26042485d2be.png)

## Pipeline

We will use a class to test the effect of different data transformations:

```python

# Custom Transformer that extracts columns passed as argument to its constructor 
class FeatureSelector( BaseEstimator, TransformerMixin ):
    # Class Constructor 
    def __init__( self, feature_names ):
        self.feature_names = feature_names 
    
    # Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    # Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self.feature_names ]
# Transform numerical features
class NumericalTransformer( BaseEstimator, TransformerMixin ):
  # Class constructor method that takes a model parameter as argument
  # model 0: minmax
  # model 1: standard
  # model 2: without scaler
  def __init__(self, model = 2):
    self.model = model
    self.colnames = None
# Return self nothing else to do here    
  def fit( self, X, y = None ):
    return self
# Return columns names after transformation
  def get_feature_names(self):
        return self.colnames 
        
  # Transformer method we wrote for this transformer 
  def transform(self, X , y = None ):
    df = X.copy()
    
    # Update columns name
    self.colnames = df.columns.tolist()
    
    if self.model == 0: 
      scaler = MinMaxScaler()
      df = scaler.fit_transform(df)
    elif self.model == 1:
      scaler = StandardScaler()
      df = scaler.fit_transform(df)
    else:
      df = df.values
return df

```
## Train-Test split

Let’s split our data.

```python

x_train, x_test, y_train, y_test = train_test_split(data.drop(labels = "diagnosis", axis = 1),
data["diagnosis"],
test_size = 0.20,
random_state = 5,
shuffle = True,
stratify = data["diagnosis"])

```
## Creating The Pipeline

```python

# Numerical features
numerical_features = x_train.select_dtypes("float64").columns.to_list()
# Defining the steps in the numerical pipeline
numerical_pipeline = Pipeline(steps = [('num_selector', FeatureSelector(numerical_features)),
('num_transformer', NumericalTransformer())])

```

## Algorithm Tuning

We evaluate 4 algorithms: KNN(K-Nearest Neighbors), Decision Tree, Random Forest, and XGBoost.

```python

# Global variables
seed = 5
num_folds = 10
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
# The full pipeline
pipe = Pipeline(steps = [("numerical_pipeline", numerical_pipeline),
                         ("fs", SelectKBest()),
                         ("clf", XGBClassifier())])
# Create a dictionary with the hyperparameters
search_space = [{"clf": [XGBClassifier()],
                 "clf__n_estimators": range(50, 500, 50),
                 "clf__max_depth": [3, 6, 8, 10, 15],
                 "clf__learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                 "clf__subsample": np.arange(0.1, 1.0, 0.1),
                 "clf__colsample_bytree": np.arange(0.1, 1.0, 0.1),
                 "clf__tree_method": ["auto", "gpu_hist"],
                 "fs__score_func": [f_classif, mutual_info_classif, chi2],
                 "fs__k": range(5, 30, 5),
                 "numerical_pipeline__num_transformer__model": range(0, 2, 1)},
                {"clf": [DecisionTreeClassifier()],
                 "clf__criterion": ["gini", "entropy"],
                 "clf__splitter": ["best"],
                 "fs__score_func": [f_classif, mutual_info_classif, chi2],
                 "fs__k": range(5, 30, 5),
                 "numerical_pipeline__num_transformer__model": range(0, 2, 1)},
                {"clf": [KNeighborsClassifier()],
                 "clf__n_neighbors": range(3, 10, 1),
                 "fs__score_func": [f_classif, mutual_info_classif, chi2],
                 "fs__k": range(5, 30, 5),
                 "numerical_pipeline__num_transformer__model": range(0, 2, 1)},
                {"clf": [RandomForestClassifier()],
                 "clf__n_estimators": range(50, 500, 50),
                 "clf__max_depth": [3, 6, 8, 10, 15],
                 "clf__criterion": ["gini", "entropy"],
                 "fs__score_func": [f_classif, mutual_info_classif, chi2],
                 "fs__k": range(5, 30, 5),
                 "numerical_pipeline__num_transformer__model": range(0, 2, 1)},
                ]
# Create grid search
kfold = StratifiedKFold(n_splits = num_folds, random_state = seed, shuffle = True)
##grid = GridSearchCV(estimator = pipe, 
                    #param_grid = search_space,
                    #cv = kfold,
                    #scoring = scoring,
                    #return_train_score = True,
                    #n_jobs = -1,
                    #refit = "Accuracy")
grid = RandomizedSearchCV(estimator = pipe, 
                    param_distributions = search_space,
                    n_iter = 150000,
                    cv = kfold,
                    scoring = scoring,
                    return_train_score = True,
                    n_jobs = -1,
                    refit = "Accuracy")
# Fit grid search
all_models = grid.fit(x_train,y_train)
print("Best: %f using %s" % (all_models.best_score_, all_models.best_params_))

```

In this example we use random search to find the best model and hyperparameter values. Random Search selects random combinations of the hyperparameter to train the model and comput the score. You can change n_iter value with the number of iterations you want to. More iterations means more computational time.

You also can use Grid Search instead of Random Search. Grid Search will run every combination of hyperparameter values, train a model and compute the desired metric. It’s computationally expensive. For the XGBClassifier hyperparameters described in the search space, the number of combinations is 10 n_estimators * 5 max_depths * 6 learning_rates… = 3,240,000. If each XGBClassifier model takes 1 second to be trained, these combinations require 37.5/n_jobs days to be computed. A google colab instance has 2 cores available, taking nearly 19 days to finish this task. The code below shows how to use Grid Search.

```python

grid = GridSearchCV(estimator = pipe, 
                    param_grid = search_space,
                    cv = kfold,
                    scoring = scoring,
                    return_train_score = True,
                    n_jobs = -1,
                    refit = "Accuracy")
                    
```

## Checking the model with best score

```python

print("Best: %f using %s" % (all_models.best_score_, all_models.best_params_))
output:
Best: 0.988425 using {'numerical_pipeline__num_transformer__model': 1, 'fs__score_func': <function f_classif at 0x7f3988f1b790>, 'fs__k': 25, 'clf__tree_method': 'auto', 'clf__subsample': 0.5, 'clf__n_estimators': 400, 'clf__max_depth': 3, 'clf__learning_rate': 0.2, 'clf__colsample_bytree': 0.1, 'clf': XGBClassifier(base_score=None, booster=None, colsample_bylevel=None,
              colsample_bynode=None, colsample_bytree=0.1, gamma=None,
              gpu_id=None, importance_type='gain', interaction_constraints=None,
              learning_rate=0.2, max_delta_step=None, max_depth=3,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=400, n_jobs=None, num_parallel_tree=None,
              random_state=None, reg_alpha=None, reg_lambda=None,
              scale_pos_weight=None, subsample=0.5, tree_method='auto',
              validate_parameters=None, verbosity=None)}
```

## Evaluating the model with Test dataset

```python

predict = all_models.predict(x_test)
confusion_matrix(predict, y_test,labels = [1, 0])
print(accuracy_score(y_test, predict))
print(classification_report(y_test, predict))
output:
array([[35,  1],
       [ 2, 70]]
0.9722222222222222
              precision    recall  f1-score   support

           0       0.97      0.99      0.98        71
           1       0.97      0.95      0.96        37

    accuracy                           0.97       108
   macro avg       0.97      0.97      0.97       108
weighted avg       0.97      0.97      0.97       108

```

## Plotting The Confusion Matrix

```python

fig, ax = plt.subplots(1, 1, figsize = (7, 4))
ConfusionMatrixDisplay(confusion_matrix(predict, y_test, labels = [1, 0]),
                       display_labels = ["Malignant", "Benign"]).plot(values_format = ".0f", ax = ax)
ax.set_xlabel("True Label")
ax.set_ylabel("Predicted Label")
plt.show()
output:

```

![confusion_matrix_medium](https://user-images.githubusercontent.com/40517814/187781577-ba414fc0-1ad1-4193-a0b4-74a2a67c32ba.png)


## Plotting The Feature contribution

```python

fig, ax = plt.subplots(1, 1, figsize = (12, 8))
xticks = [x for x in range(len(features.scores_))] 
ax.bar(xticks, features.scores_)
ax.set_xticks(xticks)
ax.set_xticklabels(features_num.get_params()["num_transformer"].get_feature_names(), rotation = 90)
ax.set_title("Feature Importance using Bestkselect")
plt.show()
output:

```
![select_k_best](https://user-images.githubusercontent.com/40517814/187781911-ce811e6c-177a-47ef-9528-be6f8413d342.png)

## Results

The best algorithm was XGBoost with accuracy of 0.988425 (with Train dataset) and 0.9722222222222222 (with Test dataset). Let’s take a look at the best model and hyperparameter using the code below.

## Checking Best Model Hyperparameters

```python

classifier = all_models.best_estimator_.named_steps['clf']
classifier
output:
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.2, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=400, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=0.5,
              tree_method='auto', validate_parameters=1, verbosity=None)
              
```

So… What you waiting for ? Use what you learned here and build your own pipeline!!
