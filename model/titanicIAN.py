from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import seaborn as sns
import numpy as np

# Define the TitanicRegression global variable
titanic_regression = None

# Define the TitanicRegression class
class Titanic2Regression:
    def __init__(self):
        self.dt = None
        self.logreg = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.encoder = None

    def initTitanic(self):
        titanic_data = sns.load_dataset('titanic')
        global td
        td = titanic_data
        td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
        td.dropna(inplace=True) # drop rows with at least one missing value, after dropping unuseful columns
        td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
        td['alone'] = td['alone'].apply(lambda x: 1 if x == True else 0)

        # Encode categorical variables
        global enc
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(td[['embarked']])
        onehot = enc.transform(td[['embarked']]).toarray()
        cols = ['embarked_' + val for val in enc.categories_[0]]
        td[cols] = pd.DataFrame(onehot)
        td.drop(['embarked'], axis=1, inplace=True)
        td.dropna(inplace=True) # drop rows with at least one missing value, after preparing the data


    def runDecisionTree(self):
        # Build distinct data frames on survived column
        X = td.drop('survived', axis=1) # all except 'survived'
        y = td['survived'] # only 'survived'

        # Split arrays in random train 70%, random test 30%, using stratified sampling (same proportion of survived in both sets) and a fixed random state (42
        # The number 42 is often used in examples and tutorials because of its cultural significance in fields like science fiction (it's the "Answer to the Ultimate Question of Life, The Universe, and Everything" in The Hitchhiker's Guide to the Galaxy by Douglas Adams). But in practice, the actual value doesn't matter; what's important is that it's set to a consistent value.
        # X_train is the DataFrame containing the features for the training set.
        # X_test is the DataFrame containing the features for the test set.
        # y-train is the 'survived' status for each passenger in the training set, corresponding to the X_train data.
        # y_test is the 'survived' status for each passenger in the test set, corresponding to the X_test data.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a decision tree classifier
        dt = DecisionTreeClassifier()
        dt.fit(X_train, y_train)

        # Test the model
        y_pred = dt.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('DecisionTreeClassifier Accuracy: {:.2%}'.format(accuracy))  

    def runLogisticRegression(self):
        # Build distinct data frames on survived column
        X = td.drop('survived', axis=1) # all except 'survived'
        y = td['survived'] # only 'survived'

        # Split arrays in random train 70%, random test 30%, using stratified sampling (same proportion of survived in both sets) and a fixed random state (42
        # The number 42 is often used in examples and tutorials because of its cultural significance in fields like science fiction (it's the "Answer to the Ultimate Question of Life, The Universe, and Everything" in The Hitchhiker's Guide to the Galaxy by Douglas Adams). But in practice, the actual value doesn't matter; what's important is that it's set to a consistent value.
        # X_train is the DataFrame containing the features for the training set.
        # X_test is the DataFrame containing the features for the test set.
        # y-train is the 'survived' status for each passenger in the training set, corresponding to the X_train data.
        # y_test is the 'survived' status for each passenger in the test set, corresponding to the X_test data.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Build distinct data frames on survived column
        X = td.drop('survived', axis=1) # all except 'survived'
        y = td['survived'] # only 'survived'
        # Train a logistic regression model
        global logreg
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)

        # Test the model
        y_pred = logreg.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print('LogisticRegression Accuracy: {:.2%}'.format(accuracy))  

def initTitanic2():
    global titanic_regression
    titanic_regression = TitanicRegression()
    titanic_regression.initTitanic()
    titanic_regression.runDecisionTree()
    titanic_regression.runLogisticRegression()

# From API
def predictSurvival(passenger):
    """
    # more code here
    # interact with
    new_passenger = passenger.copy()

    # Preprocess the new passenger data
    new_passenger['sex'] = new_passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
    new_passenger['alone'] = new_passenger['alone'].apply(lambda x: 1 if x == True else 0)

    # Encode 'embarked' variable
    onehot = enc.transform(new_passenger[['embarked']]).toarray()
    cols = ['embarked_' + val for val in enc.categories_[0]]
    new_passenger[cols] = pd.DataFrame(onehot, index=new_passenger.index)
    new_passenger.drop(['name'], axis=1, inplace=True)
    new_passenger.drop(['embarked'], axis=1, inplace=True)

    # Predict the survival probability for the new passenger
    dead_proba, alive_proba = np.squeeze(logreg.predict_proba(new_passenger))

    # Print the survival probability
    print('Death probability: {:.2%}'.format(dead_proba))  
    print('Survival probability: {:.2%}'.format(alive_proba)) 
    """
    return (lambda new_passenger: [[(lambda onehot: (lambda cols: [[new_passenger.drop(['name'], axis=1, inplace=True), [new_passenger.drop(['embarked'], axis=1, inplace=True), [[print('Death probability: {:.2%}'.format(dead_proba)), [print('Survival probability: {:.2%}'.format(alive_proba)), [dead_proba, alive_proba]][-1]][-1] for (dead_proba, alive_proba) in [np.squeeze(logreg.predict_proba(new_passenger))]][0]][-1]][-1] for new_passenger[cols] in [pd.DataFrame(onehot, index=new_passenger.index)]][0])(['embarked_' + val for val in enc.categories_[0]]))(enc.transform(new_passenger[['embarked']]).toarray()) for new_passenger['alone'] in [new_passenger['alone'].apply(lambda x: 1 if x == True else 0)]][0] for new_passenger['sex'] in [new_passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)]][0])(passenger.copy())


# Sample usage without API
if __name__ == "__main__":
    # Initialize the Titanic model
    initTitanic()
    passenger = pd.DataFrame({
        'name': [''],
        'pclass': [0],
        'sex': ['female'],
        'age': [0],
        'sibsp': [0], 
        'parch': [0], 
        'fare': [512], 
        'embarked': ['S'], 
        'alone': [False]
    })
    print(predictSurvival(passenger))