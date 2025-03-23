import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

'''there is one issue with the code I cannot fix where the confusion matrix doesn't display unless I run the other code but then it merges some of the graphs.'''
'''I used the code given to us to load the data and split into binary classes, I also used the example given to us on the worksheet for the GridSearch'''
'''The code does take some time to run fully but does work'''

'''Code given to us to load in the dataset, split it into test and training sets and convert it into binary'''

# --- 1. LOADING THE DATA ---
# Make sure this path points to your unzipped "UCI HAR Dataset" folder
PATH = r"C:\Users\yohan\Documents\data\asf\UCI HAR Dataset\\"

features_path = PATH + "features.txt"
activity_labels_path = PATH + "activity_labels.txt"
X_train_path = PATH + "train/X_train.txt"
y_train_path = PATH + "train/y_train.txt"
X_test_path = PATH + "test/X_test.txt"
y_test_path = PATH + "test/y_test.txt"

# Load feature names, this appends the column index to any duplicate names.
features_df = pd.read_csv(features_path, sep=r"\s+", header=None, names=["idx", "feature"])
features_df["feature"] = features_df["feature"].astype(str) + "_" + features_df.index.astype(str)
feature_names = features_df["feature"].tolist()

# Load activity labels (mapping IDs 1-6 to string names)
activity_labels_df = pd.read_csv(activity_labels_path, sep=r"\s+", header=None, names=["id", "activity"])
activity_map = dict(zip(activity_labels_df["id"], activity_labels_df["activity"]))

# Load train/test sets
X_train = pd.read_csv(X_train_path, sep=r"\s+", header=None, names=feature_names)
y_train = pd.read_csv(y_train_path, sep=r"\s+", header=None, names=["Activity"])
X_test = pd.read_csv(X_test_path, sep=r"\s+", header=None, names=feature_names)
y_test = pd.read_csv(y_test_path, sep=r"\s+", header=None, names=["Activity"])

# Map the activity IDs to their names
y_train["Activity"] = y_train["Activity"].map(activity_map)
y_test["Activity"] = y_test["Activity"].map(activity_map)




# --- 2. CONVERT MULTI-CLASS TO BINARY ---
def to_binary_label(activity):
    if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]:
        return 1  # Active
    else:
        return 0  # Inactive

y_train["Binary"] = y_train["Activity"].apply(to_binary_label)
y_test["Binary"] = y_test["Activity"].apply(to_binary_label)

def plot(y_labels):
    activitycounts = y_labels["Activity"].value_counts()
    
    plt.figure(figsize=(8, 5))
    activitycounts.plot(kind='bar', color='skyblue')
    plt.title("Distribution of Original Activities")
    plt.xlabel("Activity")
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.show()


'''Training SVM models with th three Kernel Types'''
def SVM(X_train, y_train, X_test, y_test):
    
    '''Linear kernel'''
    linearsvm = SVC(C=0.1, kernel='linear')
    linearsvm.fit(X_train, y_train)
    linearpredt = linearsvm.predict(X_test)
    linearacc = accuracy_score(y_test, linearpredt)
    print("Linear Kernel Accuracy:", linearacc * 100)
    
    '''Polynomial kernel'''
    polysvm = SVC(C=0.1, kernel='poly', degree=3, gamma=0.01)
    polysvm.fit(X_train, y_train)
    polypredt = polysvm.predict(X_test)
    polyacc = accuracy_score(y_test, polypredt)
    print("Polynomial Kernel Accuracy:", polyacc * 100)
    
    '''RBF kernel'''
    rbfsvm = SVC(C=1, kernel='rbf', gamma=0.01)
    rbfsvm.fit(X_train, y_train)
    rbfpredt = rbfsvm.predict(X_test)
    rbfacc = accuracy_score(y_test, rbfpredt)
    print("RBF Kernel Accuracy:", rbfacc * 100)

'''Run the grid search using the code given in the problem sheet as a template'''
def gridsearch(X_train, y_train):
    
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC())
    ])

    param_grid = [
        {'svc__kernel': ['linear'], 'svc__C': [0.1, 1, 10, 100]},
        {'svc__kernel': ['poly'], 'svc__C': [0.1, 1], 'svc__degree': [2, 3], 'svc__gamma': [0.001, 0.01, 0.1]},
        {'svc__kernel': ['rbf'], 'svc__C': [0.1, 1, 10], 'svc__gamma': [0.001, 0.01, 0.1]}
        ]


    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train.values.ravel())
    
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", round(grid_search.best_score_ * 100, 2), "%")

    return grid_search.best_estimator_

''''Creates and displays a confusion matrix and classification report'''
def confusion(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Inactive", "Active"]).plot(cmap="Blues")
    
    print("Confusion Matrix:", cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Inactive", "Active"]))


'''Executes the whole thing'''
if __name__ == "__main__":

    X_train_input = X_train
    X_test_input = X_test
    y_train_binary = y_train["Binary"]
    y_test_binary = y_test["Binary"]

        # Visualizations
    plot(y_train)
    

    
    SVM(X_train_input, y_train_binary, X_test_input, y_test_binary)

 
    bestmodel = gridsearch(X_train_input, y_train_binary)

    testaccuracy = bestmodel.score(X_test_input, y_test_binary)
    print("Test Accuracy (Best Model):", round(testaccuracy * 100, 2))


    confusion(bestmodel, X_test_input, y_test_binary)











    








