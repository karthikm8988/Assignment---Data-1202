# Assignment---Data-1202

Project Title <BR>
In this project code, we evaluated the performance of Support Vector Machine (SVM) and Naive Bayes (NB) models on the drugdataset.csv.

Prerequisites<BR>
Jupyter Notebook via Anaconda to run Python

Installing <BR>
import pandas 
import pandas as pd and load the dataset<BR>
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
file_path = '/mnt/data/drugdataset.csv'
data = pd.read_csv(file_path)
