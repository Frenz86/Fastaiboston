### just to create and test the model pkl

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


# Load dataset
from sklearn.datasets import load_boston
boston = load_boston()
# Split-out validation dataset

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=667, 
                                                    shuffle=True)
classifier = LinearRegression()
classifier.fit(X_train,y_train)

#save the model to disk
joblib.dump(classifier,'BostonClassifier.pkl')

#load the model from disk
loaded_model = joblib.load('BostonClassifier.pkl')
result = loaded_model.score(X_test, y_test)
print(result)
pred = loaded_model.predict([[3,5,4,3,3,5,4,3,3,5,4,3,3]])
print(pred)