import argparse
import os
import glob
import nn
from nn import NeuralNetwork,Layer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize
from sklearn.metrics import accuracy_score,f1_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense,Activation
from PIL import Image
import pickle
import numpy as np
def my_nn(n_vec,act):
    net = NeuralNetwork()
    for i in range(len(n_vec)-1):
        if(i == len(n_vec)-1):
            net.add_layer(Layer(n_vec[i],n_vec[i+1],'softmax'))
        else:
            net.add_layer(Layer(n_vec[i],n_vec[i+1],act))
    return net
def keras_nn(n_vec,act):
    net=Sequential()
    net.add(Dense(n_vec[1], activation=act, input_dim=n_vec[0]))
    for i in range(len(n_vec)-2):
        if(i == len(n_vec)-2):
            net.add(Dense(n_vec[i+2],activation='softmax'))
        else:
            net.add(Dense(n_vec[i+2],activation=act))

    net.compile(optimizer='sgd',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

    return net
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data')
    parser.add_argument('--train-data')
    parser.add_argument('--configuration',type=str)
    parser.add_argument('--dataset')
    scaler = StandardScaler()
    args=parser.parse_args()
    mode='test'
    if(args.dataset == 'Cat-Dog'):
        n_classes=2
    else:
        n_classes=10
    if (args.train_data != None):
        mode='train'
    #load test data
    subfolders = [f.path for f in os.scandir(args.test_data) if f.is_dir()]
    for i in range(len(subfolders)):
        imgs=[]
        for file in glob.glob(subfolders[i]+'/*.jpg'):
            img=Image.open(file).convert('L')
            img=img.resize((28,28),Image.ANTIALIAS)
            imgs.append(np.array(img).flatten())
        images=np.vstack(imgs)
        #print(images.shape)
        labels = np.array([i] * images.shape[0],dtype='int8')
        if(i==0):
            X_test=images
            y_test=labels
        else:
            X_test=np.append(X_test,images,axis=0)
            y_test=np.append(y_test,labels,axis=0)
    print(X_test.shape)
    input_dim=X_test.shape[1]
    X_test=scaler.fit_transform(X_test)
    fname='model_'+args.dataset+'.sav' 
    if (mode =='test'):
        network=pickle.load(open(fname, 'rb'))
        y_pred=network.predict(X_test,y_test)
        acc=accuracy_score(y_test,y_pred)
        f1_mic=f1_score(y_test,y_pred,average='micro')
        f1_mac=f1_score(y_test,y_pred,average='macro')
        print("Test Accuracy ::",acc)
        print("Test Macro F1-score ::",f1_mac)
        print("Test Micro F1-score ::",f1_mic)

    if (mode =='train'):
        #max_train=15000
        input_conf=[50,20]
        input_dim=784
        if(args.configuration != None):
            config = [int(item) for item in args.configuration.split(',')]
            input_conf=np.array(config)
        n_vec=[0] * (len(input_conf)+2)
        n_vec[0]=input_dim
        n_vec[1:-1]=input_conf
        n_vec[-1]=n_classes
        subfolders = [f.path for f in os.scandir(args.train_data) if f.is_dir()]
        for i in range(len(subfolders)):
            imgs=[]
            for file in glob.glob(subfolders[i]+'/*.jpg'):
                img=Image.open(file).convert('L')
                img=img.resize((28,28),Image.ANTIALIAS)
                imgs.append(np.array(img).flatten())
            images=np.vstack(imgs)
            print(images.shape)
            labels = np.array([i] * images.shape[0],dtype='int8')
            if(i==0):
                X_train=images
                y_train=labels
            else:
                X_train=np.append(X_train,images,axis=0)
                y_train=np.append(y_train,labels,axis=0)

        print(X_train.shape,y_train.shape)
        X_train, y_train = shuffle(X_train, y_train)
        X_train=scaler.fit_transform(X_train)

        #define the neural net
        #parameters
        minibatch_size=100
        learning_rate=1e-2
        max_epochs=30
        network=my_nn(n_vec,'sigmoid')
        network=network.fit(X_train,y_train, n_classes, minibatch_size, 10, max_epochs)
        save=True
        if(save==True):
            pickle.dump(network, open(fname, 'wb'))
        print("training finished")
        y_pred=network.predict(X_test,y_test)
        acc=accuracy_score(y_test,y_pred)
        f1_mic=f1_score(y_test,y_pred,average='micro')
        f1_mac=f1_score(y_test,y_pred,average='macro')
        print("Test Accuracy ::",acc)
        print("Test Macro F1-score ::",f1_mac)
    keras=False
    if(keras == True):
        onehot=np.eye(n_classes)[y_train].reshape((y_train.shape[0],n_classes))
        k_net=keras_nn(n_vec,'sigmoid')
        k_net.fit(X_train[:max_train,:], onehot[:max_train], epochs=max_epochs,batch_size=minibatch_size)
        loss,acc=k_net.evaluate(X_train[max_train:,:],onehot[max_train:])
        print(loss,acc)
