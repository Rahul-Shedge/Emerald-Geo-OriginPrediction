import pandas as pd
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from Customfuction import functions


data = pd.read_excel("Data/Emerald Origin.xlsx", header=0)

Func=functions.Function()
data=Func.conToFloat(data)
X_train, X_test, Y_train, Y_test, clsses, standScaler = Func.getSplitData(data,"ORIGIN")

Knn=KNeighborsClassifier(n_neighbors=8)
Knn.fit(X_train, Y_train)
pickle.dump(Knn, open('Models/Knn_Model.pickle','wb'))
pickle.dump(standScaler, open("Models/StandardScaler.pickle", 'wb'))


Load_Model = pickle.load(open('Models/Knn_Model.pickle', 'rb'))
standScaler = pickle.load(open('Models/StandardScaler.pickle', 'rb'))

Values = [423,19.8,133,1960,11,6320,2.47,27.1,17.2,12.5,73.8,941]
Arrays = np.array(Values).reshape(1, -1)
Scaled = standScaler.transform(Arrays)
prediction = Load_Model.predict(Scaled)[0]
num = [i for i in range(8)]
klasses = ['Colombia', 'Brazil',
           'Zambia',
           'Madagascar',
           'Afghanistan',
           'Zimbabwe',
           'Ethopia',
           'Russia']
dic = dict(zip(num, klasses))
output = [k for i, k in dic.items() if i == prediction][0]

print(output)