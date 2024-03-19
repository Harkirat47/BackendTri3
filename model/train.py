import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import seaborn as sns

class TitanicModel:
    def __init__(self):
        self.logreg = None
        self.dt = None
        self.encoder = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def train(self):
        # Load Titanic dataset
        titanic_data = sns.load_dataset('titanic')
        titanic_data.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
        titanic_data.dropna(inplace=True)
        titanic_data['sex'] = titanic_data['sex'].apply(lambda x: 1 if x == 'male' else 0)
        titanic_data['alone'] = titanic_data['alone'].apply(lambda x: 1 if x == True else 0)

        # Encode categorical variables
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        self.encoder.fit(titanic_data[['embarked']])
        onehot = self.encoder.transform(titanic_data[['embarked']]).toarray()
        cols = ['embarked_' + val for val in self.encoder.categories_[0]]
        titanic_data[cols] = pd.DataFrame(onehot)
        titanic_data.drop(['embarked'], axis=1, inplace=True)

        # Split data into train and test sets
        X = titanic_data.drop('survived', axis=1)
        y = titanic_data['survived']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train logistic regression model
        self.logreg = LogisticRegression()
        self.logreg.fit(self.X_train, self.y_train)

        # Train decision tree classifier
        self.dt = DecisionTreeClassifier()
        self.dt.fit(self.X_train, self.y_train)

    def get_models(self):
        return self.logreg, self.dt, self.encoder, self.X_train, self.X_test, self.y_train, self.y_test

# Instantiate TitanicModel and train the models
titanic_model = TitanicModel()
titanic_model.train()