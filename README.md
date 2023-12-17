# Introduction:
The detection of anomalies, especially in the context of smart home environments, is of significant importance for ensuring the safety and well-being of elderly individuals. The dataset used in this thesis focuses on the specific problem of detecting falls within a care-independent smart home setting. Falls are considered critical events that require immediate attention and assistance to prevent further injury or complications.

### Problem:
The main problem addressed in this dataset is the classification of events as either normal living activities or falling events. The goal is to develop a robust algorithm that can accurately identify instances of falls based on the sensor data collected from various locations on the person's body, such as the chest, ankles, and belt. By detecting falls in real-time, appropriate measures can be taken to provide timely assistance to the individual, potentially saving lives and reducing the risks associated with falls.

### Import important libraries

**numpy**: It is a fundamental package for scientific computing in Python. It provides support for large, multi-dimensional arrays and various mathematical functions.

**pandas**: It is a powerful data manipulation and analysis library. It provides data structures such as DataFrames that allow you to efficiently work with structured data.

**seaborn**: It is a data visualization library based on matplotlib. It provides a high-level interface for creating attractive and informative statistical graphics.

**StandardScaler**: It is a preprocessing class from scikit-learn (imported as sklearn). It is used for standardizing features by removing the mean and scaling to unit variance. This step is often performed before training machine learning models to ensure that features are on a similar scale.

**IsolationForest**: It is an anomaly detection algorithm based on the concept of isolation. It creates random forests to isolate observations that are considered as anomalies. It is a part of the scikit-learn library.

```

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for data visualization 
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
```

### Prepare the data 

```

#train
data_0 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/train/data_0.csv')
data_1 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/train/data_1.csv')
data_2 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/train/data_12.csv')
data_3 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/train/data_13.csv')
data_4 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/train/data_14.csv')
data_5 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/train/data_15.csv')
data_6 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/train/data_16.csv')
data_7 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/train/data_17.csv')
data_8 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/train/data_18.csv')
data_9 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/train/data_19.csv')

```

All of the loaded CSV files (**data_0** to **data_9**) represent the training data. Each file corresponds to data from a different person observed during the experiments. These files contain the training examples that will be used to train an anomaly detection model.

```

#test 
data_20 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/test/data_20.csv')
data_21 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/test/data_21.csv')
data_22 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/test/data_22.csv')
data_23 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/test/data_23.csv')
data_24 = pd.read_csv('/kaggle/input/anomaly-detection-falling-people-events/data/test/data_24.csv')

```

 The test data from separate CSV files (**data_20** to **data_24**). Each file corresponds to data from a different person observed during the experiments. These files contain the test examples that will be used to evaluate the performance of the anomaly detection model trained on the training data.

Check for null values in each of the train dataset and test dataset . The isna() function is applied to each dataset, which returns a boolean dataframe indicating whether each value is null or not. The sum() function is then used to calculate the sum of null values for each column in the dataframe.

By printing the sum of null values for each dataset, you can identify if there are any missing values in the data that need to be handled before further analysis or modeling.

```

# check for the nulls  for train data 
print(f"Sum of the na value in data_0: {data_0.isna().sum()}")
print(f"Sum of the na value in data_1: {data_1.isna().sum()}")
print(f"Sum of the na value in data_2: {data_2.isna().sum()}")
print(f"Sum of the na value in data_3: {data_3.isna().sum()}")
print(f"Sum of the na value in data_4: {data_4.isna().sum()}")
print(f"Sum of the na value in data_5: {data_5.isna().sum()}")
print(f"Sum of the na value in data_6: {data_6.isna().sum()}")
print(f"Sum of the na value in data_7: {data_7.isna().sum()}")
print(f"Sum of the na value in data_8: {data_8.isna().sum()}")
print(f"Sum of the na value in data_9: {data_9.isna().sum()}")

#test
print(f"Sum of the na value in data_20: {data_20.isna().sum()}")
print(f"Sum of the na value in data_21: {data_21.isna().sum()}")
print(f"Sum of the na value in data_22: {data_22.isna().sum()}")
print(f"Sum of the na value in data_23: {data_23.isna().sum()}")
print(f"Sum of the na value in data_24: {data_24.isna().sum()}")

```

Than check for duplication 


```
#check for duplicate value for train
print("Duplicated value for data_0:", data_0.duplicated().sum())
print("Duplicated value for data_1:", data_1.duplicated().sum())
print("Duplicated value for data_2:", data_2.duplicated().sum())
print("Duplicated value for data_3:", data_3.duplicated().sum())
print("Duplicated value for data_4:", data_4.duplicated().sum())
print("Duplicated value for data_5:", data_5.duplicated().sum())
print("Duplicated value for data_6:", data_6.duplicated().sum())
print("Duplicated value for data_7:", data_7.duplicated().sum())
print("Duplicated value for data_8:", data_8.duplicated().sum())
print("Duplicated value for data_9:", data_9.duplicated().sum())

#check for duplicate value for test
print("Duplicated value for data_20:", data_20.duplicated().sum())
print("Duplicated value for data_21:", data_21.duplicated().sum())
print("Duplicated value for data_22:", data_22.duplicated().sum())
print("Duplicated value for data_23:", data_23.duplicated().sum())
print("Duplicated value for data_24:", data_24.duplicated().sum())

```

Checking for duplicated values is important to ensure data quality and identify any potential issues with the dataset. Duplicated values can affect the accuracy of models and lead to biased results. By identifying and handling duplicated values appropriately, you can ensure the integrity and reliability of your data analysis.

- Concat all the data_0 to data_9 into single one called train_data, concat all the data_20 to data24

```

#concat the train
train_data = pd.concat([data_0, data_1, data_2, data_3, data_4, data_5,
                       data_6, data_7, data_8, data_9])

train_data.head()


#contcat the test 
test_data = pd.concat([data_20, data_21, data_22, data_23, data_24])

test_data.head()

```


The updated code concatenates the test data as well, creating a dataframe called test_data. Here's the modified code:

Now, both the train and test data are concatenated using the pd.concat() function. The train data is concatenated from data_0 to data_9, and the test data is concatenated from data_20 to data_24. By concatenating the data, you create two separate dataframes, train_data and test_data, containing all the rows from the individual dataframes combined into single dataframes.

```
#feature, lable
X_train = train_data[['x', 'y']]
X_test = test_data[['x', 'y']]

```

```
#convert the train_data and test_data to numpy array 
train_array = np.array(train_data)
test_array = np.array(test_data)

```

```
scaler = StandardScaler()
train_array_transformed = scaler.fit_transform(train_array)
test_array_transformed = scaler.transform(test_array)

```

Instantiates an Isolation Forest model with a contamination parameter of 0.05.

Isolation Forest is an anomaly detection algorithm that works by isolating anomalies in the data. The contamination parameter determines the expected proportion of anomalies in the data. In this case, it is set to 0.05, indicating that approximately 5% of the data is expected to be anomalous.

By setting up the Isolation Forest model with the desired contamination level, you can use it to detect anomalies in your data.

```
plt.hist(train_array_transformed[:, 0], bins=50, label='Train')
plt.hist(test_array_transformed [:, 0], bins=50, label='Test')
plt.xlabel('Feature 1')
plt.ylabel('Frequency')
plt.title('Histogram of Feature 1')
plt.legend()
plt.show()
```

Than we move to creating the model

```

model = IsolationForest(contamination=0.05)
model.fit(train_array_transformed)

```
This is fitting an Isolation Forest anomaly detection model on some training data:

IsolationForest - Creates a new IsolationForest model object. The contamination parameter specifies the proportion of outliers expected in the training data. 0.05 means the model expects 5% of the data to be anomalies.
fit() - Fits the Isolation Forest model to the training data. train_array_transformed is presumably a NumPy array or Pandas DataFrame containing the numeric feature data to train the model on. This will analyze the data to determine normal vs anomalous points, setting internal thresholds.

Once the model is fitted, it can be used to make predictions or detect anomalies in new, unseen data (test_data).

```
pred_test = model.predict(test_array_transformed)
print(pred_test)

```

Than transforming the predictions from the model (stored in pred_test) into 0s and 1s by using np.where().

Specifically:

pred_test - This is an array containing predictions, with -1 indicating a prediction of the normal class, and 1 indicating a prediction of the anomaly class.
np.where() - Applies a mapping conditional on the values in pred_test. For each value:
If it's -1, replace it with 0
Else (meaning 1), replace it with 1
pred_test_mapped - The output array with the mapped values.
So now instead of -1 and 1 values, pred_test_mapped contains 0 to indicate normal predictions and 1 to indicate anomalies. This binary encoding can be useful for evaluating the model performance later.

```
# Mapping -1 to 0 and 1 to 1 in pred_test
pred_test_mapped = np.where(pred_test == -1, 0, 1)

#checking for th tranformation 
set(pred_test_mapped)
```
In essence, it's transforming the raw anomaly model predictions into a more intuitive binary classification output.

The predict() function of the Isolation Forest model is used to predict whether each sample in the test data is an outlier (anomaly) or not. It assigns an anomaly score to each sample, where a negative score indicates an outlier and a positive score indicates an inlier.

the decision_function() method of the Isolation Forest model is used to obtain the anomaly scores for the test data.

The decision_function() method calculates the anomaly score for each sample in the test data. The anomaly score represents the degree of abnormality of each sample, where a higher score indicates a higher likelihood of being an anomaly.

The scores variable stores the anomaly scores calculated by the decision_function() method for each sample in the test data. The values in the scores array can be interpreted as follows:

Negative scores indicate that the corresponding samples are more likely to be anomalies.
Positive scores indicate that the corresponding samples are more likely to be inliers (not anomalies).
Printing scores will display the anomaly scores for each sample in the test data, providing information about their abnormality levels.






