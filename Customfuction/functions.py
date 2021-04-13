import pandas as pd
import numpy as np
from application_logging import logger
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split



class Function:

    def __init__(self):
        self.file_object = open("Training_Logs/Training_logs.txt",'a+')
        self.log_writer = logger.App_Logger()

    def conToFloat(self,Data):
        try:
            coln = list(Data.columns[1:-2])
            for i in coln:
                Data[i] = pd.to_numeric(Data[i], errors='coerce')
                Data = Data.replace(np.nan, 0, regex=True)
                Data[i] = Data[i].astype(float)
            self.log_writer.log(self.file_object, 'conToFloat function executed successfully')
        except Exception as ex:
            self.log_writer.log(self.file_object,'Error occured while running conToFloat function!! Error:: %s' % ex)
            raise ex
        return Data.iloc[:, 1:]

    def getSplitData(self,Data, OutputLabel):
        try:
            le = LabelEncoder()
            sc = StandardScaler()
            train, test = train_test_split(Data, test_size=0.2, random_state=5)
            Y_train = le.fit_transform(train[str(OutputLabel)])
            Y_test = le.transform(test[str(OutputLabel)])
            X_train = train.drop(str(OutputLabel), axis=1)
            X_test = test.drop(str(OutputLabel), axis=1)
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            cls = list(le.classes_)
            clsses = [x.split(",")[1].lstrip() for x in cls]
            self.log_writer.log(self.file_object, 'getSplitData function worked well.')
        except Exception as ex:
            self.log_writer.log(self.file_object,'Error occured while running getSplitData function!! Error:: %s' % ex)
            raise ex
        return X_train, X_test, Y_train, Y_test, clsses, sc