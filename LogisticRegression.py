# Import necessary libraries
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Loading the Breast Cancer dataset
cancer = datasets.load_breast_cancer()

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

# Define the hyperparameters for grid search
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}

# Create a logistic regression classifier
clf = LogisticRegression()

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Train the model using the best hyperparameters
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate the model using k-fold cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Cross-validation scores: ", cv_scores)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Calculating the accuracy, sensitivity, and specificity of the model on the test data
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print the results
print("Accuracy: ", acc)
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)
