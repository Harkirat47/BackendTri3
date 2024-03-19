import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns

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

        # Prepare response
passenger = pd.DataFrame({
    'name': ['Mary Marrry'],
    'pclass': [1], # 2nd class picked as it was median, bargains are my preference, but I don't want to have poor accomodations
    'sex': ['female'],
    'age': [64],
    'sibsp': [1], # I usually travel with my wife
    'parch': [1], # currenly I have 1 child at home
    'fare': [83.00], # median fare picked assuming it is 2nd class
    'embarked': ['S'], # majority of passengers embarked in Southampton
    'alone': [False] # travelling with family (spouse and child))
})

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