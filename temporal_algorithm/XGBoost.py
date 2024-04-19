import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data = np.load('dataset_normalized.npz', allow_pickle=True)
# X = data['array1']
X = data['array1'].reshape(889, 5 * 20)
y = data['array2']

print(X.shape)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Instantiate an XGBoost classifier object
xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=3)

# Fit the classifier to the training set
xgb_clf.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = xgb_clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))
