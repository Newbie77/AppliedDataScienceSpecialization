# author: Asmaa ~ 2019
# Coursera: Applied Machine Learning
# WEEK3 - Evaluation - Classification Evaluation Metrics

# import libraries
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# CONFUSION MATRICES

# load some data
digit_data = load_digits()
X, y = digit_data.data, digit_data.target

# make data imbalanced (edit outputs)
y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0
X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)

# use dummy classifier to fit the model by most frequent strategy
majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)

# predict outputs by the new model
y_majority_p = majority.predict(X_test)

# ----------------------------------------------------------------------

# TRY DIFFERENT SITUATIONS OF CONFUSION MATRICES 

# calculate confusion matrix 
confusion = confusion_matrix(y_test, y_majority_p)
print('Most frequent class:', confusion)

# use dummy classifier to fit the model by most frequent strategy
strat = DummyClassifier(strategy='stratified').fit(X_train, y_train)
y_strat_p = strat.predict(X_test)
confusion = confusion_matrix(y_test, y_strat_p)

print('Random class-proportional prediction:', confusion)

# create logistic regression model
logR = LogisticRegression().fit(X_train, y_train)
logR_p = logR.predict(X_test)

# test confusion matrix on logistic regression model
confusion = confusion_matrix(y_test, logR_p)
print('Logistic regression:', confusion)

# ----------------------------------------------------------------------

# EVALUATION METRICS

# Formulas:
# -----------
# Accuracy = TP + TN / (TP + TN + FP + FN)
# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate
# F1 = 2 * Precision * Recall / (Precision + Recall)

print('Accuracy:', accuracy_score(y_test, logR_p))
print('Precision:', precision_score(y_test, logR_p))
print('Recall:', recall_score(y_test, logR_p))
print('F1:', f1_score(y_test, logR_p))

# whole report about the results
print(classification_report(y_test, logR_p, target_names=['not 1', '1']))

# ----------------------------------------------------------------------

# MULTI-CLASS CONFUSION MATRIX
# use digit dataset
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(X, y, random_state=0)

# fit the model using svm with linear kernel
svm = SVC(kernel = 'linear').fit(X_train_mc, y_train_mc)

# predict using the new model
svm_p_mc = svm.predict(X_test_mc)

# calculate confusion matrix
conf_mc = confusion_matrix(y_test_mc, svm_p_mc)

# prepare confusion matrix for visualization
df_cm = pd.DataFrame(conf_mc, index = [i for i in range(0,10)], columns = [i for i in range(0,10)])
plt.figure(figsize=(5.5,4))
sns.heatmap(df_cm, annot=True)
plt.title('SVM Linear Kernel, Accuracy:{0:.3f}'.format(accuracy_score(y_test_mc, svm_p_mc)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

# get whole report
print(classification_report(y_test_mc, svm_p_mc))
