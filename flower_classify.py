import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

data = pd.read_csv("Iris.csv")

data.head()
data['Species'].value_counts()

X = data.iloc[ : ,1:5]
y = data.iloc[ : ,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
model.predict(np.array([[2,3,4,5]]))


import pickle
pickle.dump(model, open("flower-v1.pkl", "wb"))
model_pk = pickle.load(open("flower-v1.pkl", "rb"))
model_pk.predict(np.array([[2,3,4,5]]))

#from sklearn.externals import joblib
#joblib.dump(model, open("flower-v1.jl", "wb"))
#model_jl = joblib.load(open("flower-v1.pkl", "rb"))
#model_jl.predict(np.array([[2,3,4,5]]))
