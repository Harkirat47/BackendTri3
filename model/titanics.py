import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns

# define titanic regression
titanic_regression = None
# class
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
                global td
                self.td = titanic_data
                self.td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
                self.td.dropna(inplace=True) # drop rows with at least one missing value, after dropping unuseful columns
                self.td['sex'] = self.td['sex'].apply(lambda x: 1 if x == 'male' else 0)
                self.td['alone'] = self.td['alone'].apply(lambda x: 1 if x == True else 0)
            
        # Encode categorical variables
                self.enc = OneHotEncoder(handle_unknown='ignore')
                self.enc.fit(self.td[['embarked']])
                self.onehot = self.enc.transform(self.td[['embarked']]).toarray()
                cols = ['embarked_' + val for val in self.enc.categories_[0]]
                self.td[cols] = pd.DataFrame(self.onehot)
                self.td.drop(['embarked'], axis=1, inplace=True)
                self.td.dropna(inplace=True)

        # Split arrays or matrices into random train and test subsets
        def runDecisionTree(self):
                X = self.td.drop('survived', axis=1)
                y = self.td['survived']
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                dt = DecisionTreeClassifier()
                dt.fit(self.X_train, self.y_train)
                self.dt = dt

        # Train a logistic regression model
        def runLogisticRegression(self, X, y):
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                self.logreg = LogisticRegression()
                self.logreg.fit(self.X_train, self.y_train)
            
        # get the passenger from frontend
        def predict(self,data):
                passenger = data.get("passenger")
                new_passenger = {
                        'name': passenger.get('name'),
                        'pclass': passenger.get('pclass'),
                        'sex': passenger.get('sex'),
                        'age': passenger.get('age'),
                        'sibsp': passenger.get('sibsp'),
                        'parch': passenger.get('parch'),
                        'fare': passenger.get('fare'),
                        'embarked': passenger.get('embarked'),
                        'alone': passenger.get('alone'),
                }
                new_passenger['sex'] = new_passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
                new_passenger['alone'] = new_passenger['alone'].apply(lambda x: 1 if x == True else 0)
                # Encode 'embarked' variable
                onehot = self.enc.transform(new_passenger[['embarked']]).toarray()
                cols = ['embarked_' + val for val in self.enc.categories_[0]]
                new_passenger[cols] = pd.DataFrame(onehot, index=new_passenger.index)
                new_passenger.drop(['name'], axis=1, inplace=True)
                new_passenger.drop(['embarked'], axis=1, inplace=True)
                # calculate dead/alive probability
                dead_proba, alive_proba = np.squeeze(self.logreg.predict_proba(new_passenger))
                return alive_proba

def initTitanic():
        global titanic_regression
        titanic_regression = TitanicRegression()
        titanic_regression.initTitanic()
        X = titanic_regression.td.drop('survived', axis=1)
        y = titanic_regression.td['survived']
        titanic_regression.runLogisticRegression(X, y)
