from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas
import matplotlib.pyplot as plt


from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.model_selection import KFold

from tensorflow.keras.utils import to_categorical
from collections import Counter
import sklearn
import tensorflow.keras
#from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import cross_val_score
#from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D

from tensorflow.keras.layers import Dense, Dropout, Activation, SimpleRNN,GRU
from tensorflow.keras import regularizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import *

from imblearn.under_sampling import NeighbourhoodCleaningRule
from xgboost import XGBClassifier
import autokeras as ak


def test(x,y,model):
    kfold = KFold(n_splits=10,shuffle=True,random_state=7)
    n_iter = 0
    cv_f1_score = []


    for train_idx, test_idx in kfold.split(x):


        X_train, X_test = x[train_idx], x[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train,y_train,epochs=150)

        fold_pred  = model.predict(X_test)
        n_iter += 1
        #print(y_test)
        #print(fold_pred)
        c_f1_score = np.round(f1_score(y_test,fold_pred>0.5),4)
#        print(accuracy_score(y_test,fold_pred>0.5))
        print(c_f1_score)
        print('\n{} 교차검증 정확도 : {}, 학습 데이터 크기 : {}, 검증 데이터 크기 : {}'.format(n_iter,f1_score,X_train.shape[0],X_test.shape[0]))
        cv_f1_score.append(c_f1_score)

    print('\n average f1_score : ',np.mean(cv_f1_score))
    return np.mean(cv_f1_score)

def testModel(x,y,num,epochs=100,batch_size=100):
    kfold = KFold(n_splits=10,shuffle=True,random_state=7)
    n_iter = 0
    cv_f1_score = []


    for train_idx, test_idx in kfold.split(x):
        c = createModel(x,num)

        X_train, X_test = x[train_idx], x[test_idx]

        y_train, y_test = y[train_idx], y[test_idx]
        if num == 2 or num == 3 or num == 5:
            c.fit(X_train,y_train)

        else:
            c.fit(X_train,y_train,epochs=epochs)
            #history = c.fit(X_train,y_train,batch_size =batch_size,verbose=1,validation_split=0.1)

        fold_pred  = c.predict(X_test)
        n_iter += 1
        #print(y_test)
        #print(fold_pred)
        c_f1_score = np.round(f1_score(y_test,fold_pred>0.5),4)
#        print(accuracy_score(y_test,fold_pred>0.5))
        print(c_f1_score)
        print('\n{} 교차검증 정확도 : {}, 학습 데이터 크기 : {}, 검증 데이터 크기 : {}'.format(n_iter,f1_score,X_train.shape[0],X_test.shape[0]))

        cv_f1_score.append(c_f1_score)
        print("현재까지의 평균",np.mean(cv_f1_score))

    print('\n average f1_score : ',np.mean(cv_f1_score))
    return np.mean(cv_f1_score)

def getResample(x,y):
    clf = RandomForestClassifier(n_estimators=300, max_depth=9,
                             random_state=0)

    clf.fit(x, y)
    #clf.feature_importances_
    model = SelectFromModel(clf, prefit=True)
    x = model.transform(x)
    ncr = NeighbourhoodCleaningRule()
    x_resampled, y_resampled = ncr.fit_resample(x, y)
    return x_resampled, y_resampled


def createModel(x,num):
    # 0.812
    if num == 0:
        model = Sequential()
        opt=RMSprop(lr=0.00014, rho=0.9, epsilon=None, decay=0.0)
        model.add(Dense(128, input_dim=x.shape[1], activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.3, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.3, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.3, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.3, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.3, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.3, noise_shape=None, seed=None)

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

        return model
    elif num == 1: # autokeras
        #return ak.StructuredDataClassifier(max_trials=50,overwrite=False)
        return ak.StructuredDataClassifier(max_trials=50,overwrite=False)

    elif num == 2: # adaboost
        tree = DecisionTreeClassifier(max_depth=30,
                              criterion='entropy',
                              random_state=1)

        AdaBoost = AdaBoostClassifier(base_estimator=tree,
                            n_estimators=100,
                            learning_rate=0.1,
                            random_state=1)
        return AdaBoost
    elif num == 3: # Gradient Boosting
        GradientBoost = GradientBoostingClassifier(random_state=1)
        return GradientBoost
    elif num == 4:
        model = Sequential()
        model.add(LSTM(128, input_dim=x.shape[1], return_sequences=True))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    elif num == 5:
        model = XGBClassifier(n_estimators=1000,learning_rate = 0.1, max_depth=30)
        return model

    elif num == 6:
            cnn = Sequential()
            cnn.add(Conv2D(64, (3,1), padding = "same", activation = 'elu', input_dim=x.shape[1]))
            cnn.add(BatchNormalization())
            cnn.add(Conv2D(16, (2, 1), padding = "same", activation = 'elu')) # elu가  relu의 장점만 가져온 것
            cnn.add(BatchNormalization())
            cnn.add(Conv2D(64, (3, 2), padding = "same", activation = 'elu'))
            cnn.add(BatchNormalization())
            cnn.add(Conv2D(16, (2, 1), padding = "same", activation = 'elu'))
            cnn.add(BatchNormalization())
            cnn.add(Conv2D(64, (4, 3), padding = "same", activation = 'elu'))
            cnn.add(BatchNormalization())
            cnn.add(GlobalAveragePooling2D())
            cnn.add(Dense(32, activation = 'relu'))
            cnn.add(Dense(1, activation = 'sigmoid'))

    elif num == 7: # 0.849
        opt=RMSprop(lr=0.00007, rho=0.9, epsilon=None, decay=0.0)
        #Multi Layer Perceptron Model
        model = Sequential()

        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(256, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(32, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])
        return model
    elif num == 8: # 0.824
        opt=RMSprop(lr=0.00007, rho=0.9, epsilon=None, decay=0.0)
        #Multi Layer Perceptron Model
        model = Sequential()

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(128, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(32, activation='relu'))
        tensorflow.keras.layers.AlphaDropout(0.5, noise_shape=None, seed=None)

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])
        return model

    elif num == 9: # 0.822
        opt=RMSprop(lr=0.00007, rho=0.9, epsilon=None, decay=0.0)
        model = Sequential()

        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(256, activation='relu'))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(128, activation='relu'))
        model.add(tensorflow.keras.layers.AlphaDropout(0.1, noise_shape=None, seed=None))

        model.add(Dense(128, activation='relu'))
        model.add(tensorflow.keras.layers.AlphaDropout(0.1, noise_shape=None, seed=None))

        model.add(Dense(32, activation='relu'))
        model.add(tensorflow.keras.layers.AlphaDropout(0.1, noise_shape=None, seed=None))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model

    elif num == 10: # 0.815
        opt=RMSprop(lr=0.00007, rho=0.9, epsilon=None, decay=0.0)
        model = Sequential()

        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(256, activation='relu'))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(128, activation='relu'))
        model.add(tensorflow.keras.layers.AlphaDropout(0.2, noise_shape=None, seed=None))

        model.add(Dense(128, activation='relu'))
        model.add(tensorflow.keras.layers.AlphaDropout(0.2, noise_shape=None, seed=None))

        model.add(Dense(128, activation='relu'))
        model.add(tensorflow.keras.layers.AlphaDropout(0.2, noise_shape=None, seed=None))

        model.add(Dense(32, activation='relu'))
        model.add(tensorflow.keras.layers.AlphaDropout(0.2, noise_shape=None, seed=None))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model

    elif num == 11: # 0.831
        opt=RMSprop(lr=0.00007, rho=0.9, epsilon=None, decay=0.0)
        #Multi Layer Perceptron Model
        model = Sequential()
        #model.add(Dense(128, input_dim=639, activation='relu'))

        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model

    elif num == 12: #0.83
        opt=RMSprop(lr=0.00007, rho=0.9, epsilon=None, decay=0.0)
        #Multi Layer Perceptron Model
        model = Sequential()

        model.add(Dense(1024, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model
    elif num == 13: #0.833
        opt=RMSprop(lr=0.00007, rho=0.9, epsilon=None, decay=0.0)
        #Multi Layer Perceptron Model
        model = Sequential()

        model.add(Dense(1024, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model
    elif num == 14: #0.835
        opt=RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
        model = Sequential()
        model.add(Dense(1024, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(256, activation='relu'))

        model.add(Dense(128, activation='relu'))

        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))


        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))



        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model
    elif num == 15: # 0.8560000000000001
        opt=RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
        model = Sequential()

        model.add(Dense(1024, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(512, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))

        model.add(Dense(64, activation='relu'))

        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model
    elif num == 16: #0.81
        opt=RMSprop(lr=0.00007, rho=0.9, epsilon=None, decay=0.0)
        model = Sequential()

        model.add(Dense(2048, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(1024, input_dim=x.shape[1], activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model
    elif num == 17: #0.782
        opt=RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
        model = Sequential()
        model.add(Dense(128, activation='relu'))

        model.add(Dense(64, activation='relu'))

        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model
    elif num == 18: #0.814
        opt=RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
        model = Sequential()

        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))

        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model
    elif num == 19: #0.822
        opt=RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
        #Multi Layer Perceptron Model
        model = Sequential()

        model.add(Dense(1024, input_dim=x.shape[1], activation='relu'))


        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))

        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model
    elif num == 20: #0.812
        opt=RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
        #Multi Layer Perceptron Model
        model = Sequential()
        #model.add(Dense(128, input_dim=639, activation='relu'))

        model.add(Dense(2048, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(1024, input_dim=x.shape[1], activation='relu'))

        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))

        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))


        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
        return model

    else:
        return "error"
