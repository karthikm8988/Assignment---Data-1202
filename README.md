# Assignment---Data-1202<BR>
Saikrishna Mudaliar<BR>
Student Number - 100958445

<B>Project Title</B> <BR>
In this project code, we evaluated the performance of Support Vector Machine (SVM) and Naive Bayes (NB) models on the drugdataset.csv.

--------------

<B>Prerequisites</B><BR>
Jupyter Notebook via Anaconda to run Python

---------------

<B>Installing </B><BR>

import pandas 
import pandas as pd and load the dataset<BR>
Run the codes in Jupyter environment<BR>
``` Python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = '/mnt/data/drugdataset.csv'
data = pd.read_csv(file_path)
```

------------

<B>Code for using SVM and NB models</B>
```
# Preprocess the data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split the data into features and target
X = data.drop('Drug', axis=1)
y = data['Drug']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an SVM model
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Train a Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
```

-----------------

<B>Running the Tests</B><BR>
Test the coding done
```
# Evaluate the SVM model
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_classification_report = classification_report(y_test, y_pred_svm)


# Evaluate the Naive Bayes model
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_classification_report = classification_report(y_test, y_pred_nb)

```

Results expected
```
Results
SVM Model
Accuracy: 0.975

              precision    recall  f1-score   support

           0       1.00      1.00      1.00         6
           1       1.00      1.00      1.00         3
           2       1.00      0.80      0.89         5
           3       1.00      1.00      1.00        11
           4       0.94      1.00      0.97        15

    accuracy                           0.97        40
   macro avg       0.99      0.96      0.97        40
weighted avg       0.98      0.97      0.97        40

Naive Bayes Model
Accuracy: 0.9

              precision    recall  f1-score   support

           0       0.75      1.00      0.86         6
           1       0.75      1.00      0.86         3
           2       0.83      1.00      0.91         5
           3       1.00      1.00      1.00        11
           4       1.00      0.73      0.85        15

    accuracy                           0.90        40 
   macro avg       0.87      0.95      0.89        40 
weighted avg       0.92      0.90      0.90        40
```
---------------------------

<B>Break down into end to end tests</B><BR>
1. Accuracy: Accuracy is the ratio of correct predictions to total predictions made. In this case 97% and 90% respectively.<BR>
2. Precision: Precision is about being precise, i.e., how accurate your model is. In other words, you can say, when a model makes a prediction, how often 
   it is correct. In this case, SVM generally has higher precision across all classes compared to Naive Bayes. This indicates that SVM is better at 
   minimizing false positives<BR>
3. Recall: Recall is the ratio of correctly predicted positive observations to the all observations in actual class. In this case SVM also shows higher 
   recall values than Naive Bayes, meaning it is better at identifying true positives.<BR>
4. F1 score: The F1-score, which balances precision and recall, is higher for SVM across all classes, indicating a better overall performance compared to 
   Naive Bayes<BR>

-----------

<B>Built With</B> 

Jupyter from Anaconda<BR>
Excel Dataset

---------

<B> Comaprison and Insight </B>

Model Performance: SVM outperforms Naive Bayes in all key metrics (precision, recall, and F1-score). This suggests that SVM is more effective for this particular classification task. <BR>
Use Cases: SVM is preferred when the goal is to achieve high accuracy and minimize both false positives and false negatives. Naive Bayes, while slightly less accurate, can still be useful due to its simplicity and faster computation, particularly with very large datasets or when quick, approximate results are sufficient. <BR>
Overall for the data, SVM provides better classification


 
<B>Author</B>

Saikrishna Mudaliar

