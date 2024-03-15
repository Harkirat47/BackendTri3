import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns

class Titanic(passenger_data):
    def post(self):

            # Preprocess the received data
            passenger_df = pd.DataFrame(passenger_data)
            passenger_df['sex'] = passenger_df['sex'].apply(lambda x: 1 if x == 'male' else 0)
            passenger_df['alone'] = passenger_df['alone'].apply(lambda x: 1 if x else 0)

        # Load the titanic dataset
            titanic_data = sns.load_dataset('titanic')
            td = titanic_data
            td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
            td.dropna(inplace=True) # drop rows with at least one missing value, after dropping unuseful columns
            td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
            td['alone'] = td['alone'].apply(lambda x: 1 if x == True else 0)
            
        # Encode categorical variables
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(td[['embarked']])
            onehot = enc.transform(td[['embarked']]).toarray()
            cols = ['embarked_' + val for val in enc.categories_[0]]
            td[cols] = pd.DataFrame(onehot)
            td.drop(['embarked'], axis=1, inplace=True)
            td.dropna(inplace=True)

        # Split arrays or matrices into random train and test subsets
            X = td.drop('survived', axis=1)
            y = td['survived']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a logistic regression model
            logreg = LogisticRegression()
            logreg.fit(X_train, y_train)
            
        # Train a decision tree classifier
            dt = DecisionTreeClassifier()
            dt.fit(X_train, y_train)

        # Predict the survival probability for the new passenger
            dead_proba, alive_proba = np.squeeze(logreg.predict_proba(new_passenger))

        # Prepare response
            response = {
                'death_probability': round(dead_proba * 100, 2),
                'survival_probability': round(alive_proba * 100, 2)
            }
            return response