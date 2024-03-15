 ## Python Titanic Model

# Import the required libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import seaborn as sns

# Define the TitanicRegression global variable
titanic_regression = None

# Define the TitanicRegression class
class TitanicRegression:
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
        X = titanic_data.drop('survived', axis=1)
        y = titanic_data['survived']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize the encoder
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.X_train = self.encoder.fit_transform(self.X_train)
        self.X_test = self.encoder.transform(self.X_test)

        self.dt = DecisionTreeClassifier()
        self.dt.fit(self.X_train, self.y_train)

        self.logreg = LogisticRegression()
        self.logreg.fit(self.X_train, self.y_train)

    def runDecisionTree(self):
        if self.dt is None:
            print("Decision Tree model is not initialized. Please run initTitanic() first.")
            return
        y_pred_dt = self.dt.predict(self.X_test)
        accuracy_dt = accuracy_score(self.y_test, y_pred_dt)
        print('Decision Tree Classifier Accuracy: {:.2%}'.format(accuracy_dt))

    def runLogisticRegression(self):
        if self.logreg is None:
            print("Logistic Regression model is not initialized. Please run initTitanic() first.")
            return
        y_pred_logreg = self.logreg.predict(self.X_test)
        accuracy_logreg = accuracy_score(self.y_test, y_pred_logreg)
        print('Logistic Regression Accuracy: {:.2%}'.format(accuracy_logreg))

def initTitanic():
    global titanic_regression
    titanic_regression = TitanicRegression()
    titanic_regression.initTitanic()
    titanic_regression.runDecisionTree()
    titanic_regression.runLogisticRegression()

    # Store column names for reference
    titanic_regression.column_names = titanic_data.drop('survived', axis=1).columns.tolist()

def predictSurvival(passenger):
    passenger_df = pd.DataFrame(passenger, index=[0])   
    passenger_df.drop(['name'], axis=1, inplace=True)
    passenger = passenger_df.copy()

    # Add missing columns and fill them with default values
    missing_cols = set(titanic_regression.column_names) - set(passenger.columns)
    for col in missing_cols:
        passenger[col] = 0

# Sample usage
if __name__ == "__main__":
    # Initialize the Titanic model
    initTitanic()

    # Predict the survival of a passenger
    passenger = {
        'name': ['John Mortensen'],
        'pclass': [2],
        'sex': ['male'],
        'age': [64],
        'sibsp': [1],
        'parch': [1],
        'fare': [16.00],
        'embarked': ['S'],
        'alone': [False]
    }
    print(predictSurvival(passenger))