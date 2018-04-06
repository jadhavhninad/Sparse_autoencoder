# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:49:30 2018

@author: nhjadhav

Package Versions : 

Python 3.6.2 :: Continuum Analytics, Inc.
tensorflow (1.7.0)
Keras (2.1.5)

Kindly use modified file load_dataset_se provided in the submission to load mnist dataset
"""
from load_dataset_se import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import backend as K
from keras import initializers

sess = tf.Session()

def get_kld(p, p_hat):
    kl_val =  p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)
    return kl_val

def init_weights(shape):
    xinit = tf.contrib.layers.xavier_initializer()
    return tf.Variable(xinit(shape))

#=============================
#AutoEncoder Using TensorFlow
#=============================
def se_autoencoder(data,p,l,b):
    
    #Comment regarding error seen in cmd during execution
    print("\nThe below error will be fixed in tensorflow 1.7.1 version")
    
    #---------initialize the weights---------
    n_in=784
    n_out=200
    W1=init_weights((n_out,n_in))
    b1=init_weights((n_out,1))
    
    #Swap for W2 here
    W2=init_weights((n_in,n_out))
    b2=init_weights((n_in,1))
    
    activation=tf.nn.sigmoid
    optimizer=tf.train.GradientDescentOptimizer(0.1)
    lval=l
    beta=b
    X = tf.cast(data, tf.float32)
    
    #---------Generate the model-------------
    A1 = activation(tf.matmul(W1,X) + b1)
    X_hat = activation(tf.matmul(W2,A1) + b2)
    p_hat=tf.reduce_mean(A1,axis=0)
    
    
    kld=get_kld(p, p_hat)
    cost=tf.losses.mean_squared_error(X,X_hat)  + 0.5*lval*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) + beta*tf.reduce_mean(kld)
    weight_list=[W1,W2]
    train_step=optimizer.minimize(cost, var_list=weight_list)
    
    #---------train the model----------------
    init = tf.global_variables_initializer()    
    sess.run(init)  
    
    print("\n---Cost for Autoencoder during SGD------")
    for i in range(400):
        sess.run(train_step)
        if(i%100 == 0):
            print("cost at iteration", i, cost.eval(session=sess))
    
     
    W1_val = W1.eval(session=sess)    
    A1_val = A1.eval(session=sess)
    sess.close()
    return A1_val, W1_val

#==================================
#Generic Neural network using Keras
#==================================
class custom_wts(initializers.Initializer):
    def __init__(self, weight = None):
        self.W = weight

    def __call__(self, shape, dtype=None):
        return self.W
 

def keras_nn(X,y,X_test,y_test, itr,dims, wts=[], lc=0):
    nn_model = Sequential()
    
    loss_and_metrics=0
    if(lc == 2): #for training simple model using A1 from autoencoder and softmax
        nn_model.add(Dense(units=dims[1], activation='softmax', input_dim=dims[0]))
        nn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        nn_model.fit(X, y, epochs=itr, verbose=0)
    
    elif(lc == 3): #for training generic NN model
        nn_model.add(Dense(units=dims[1], activation='sigmoid', input_dim=dims[0]))
        nn_model.add(Dense(units=dims[2], activation='softmax'))
        nn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        nn_model.fit(X, y, epochs=itr, verbose=0)
        loss_and_metrics = nn_model.evaluate(X_test, y_test, verbose=0)
    
    else: #For fine-tuning combined encoder and softmax models using custom initilizer
        nn_model.add(Dense(units=dims[1], activation='sigmoid', input_dim=dims[0], kernel_initializer=custom_wts(wts[0])))
        nn_model.add(Dense(units=dims[2], activation='softmax', kernel_initializer = custom_wts(wts[1])))
        nn_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        nn_model.fit(X, y, epochs=itr, verbose=0)
        loss_and_metrics = nn_model.evaluate(X_test, y_test, verbose=0)
    
    
    for layer in nn_model.layers:
        weights = layer.get_weights()
        #print(len(weights))
    
    return loss_and_metrics, weights


#==================================
#Main Function
#==================================
def main():
    train_data = []
    train_label = []
    
    for i in range(10):
        td, tl,  =  mnist(ntrain=60000,ntest=0,digit_range=[i,i+1])
        train_data.append(td)
        train_label.append(tl)
    
    train_X = train_data[0][:,0:100]
    train_y = train_label[0][:,0:100]
    for i in range(1,10,1):
        train_X = np.concatenate((train_X, train_data[i][:,0:100]),axis=1)
        train_y = np.concatenate((train_y, train_label[i][:,0:100]),axis=1)
    
    train = np.concatenate((train_X.T,train_y.T),axis=1)
    np.random.shuffle(train)
    
    test_X = train_data[0][:,100:]
    test_y = train_label[0][:,100:]
    for i in range(1,10,1):
        test_X = np.concatenate((test_X, train_data[i][:,100:]),axis=1)
        test_y = np.concatenate((test_y, train_label[i][:,100:]),axis=1)
       
    test = np.concatenate((test_X.T,test_y.T),axis=1)
    np.random.shuffle(test)
    
    
    train_X = train[:,0:784].T
    train_y = train[:,784].T
    test_X = test[:,0:784].T
    test_y = test[:,784].T
     
    #---------Get the accuracy for genreic full-connected layer---------
    #convert labels to one_hot representation
    trY = to_categorical(train_y)
    teY = to_categorical(test_y)
    dims =[784,200,10]
    itr=400
    vals,wt = keras_nn(train_X.T, trY, test_X.T, teY, itr, dims, lc=3)
    print("--------------------------------------------------------")
    print("Generic NN: Test data loss = ", vals[0]," accuracy = ", vals[1]*100)
    
    #---------Train the autoencoder and get hidden representation---------
    p=0.1
    A1, w1 = se_autoencoder(train_X,p,0.001,3)
    #print(A1.shape)
    print("--------------------------------------------------------")
    print("Shape of W1 obtained from autoencoder = ", w1.shape)
    
    #---------Just training the compressed representation and softmax to get W3---------
    dims=[200,10]
    vals,wt = keras_nn(A1.T, trY, test_X.T, teY, itr, dims, lc=2)
    w3 = wt[0]
    print("--------------------------------------------------------")
    print("Shape of W3 obtained from model of compressed representation and softmax = ", w3.shape)
    
    #---------=FineTune the network and Get Test Accuracy for the encoder and softmax connected model---------
    dims=[784,200,10]
    #wts= [tf.cast(w1.T, tf.float32), tf.cast(w3, tf.float32) ]
    wts= [w1.T, w3 ]
    vals,wt = keras_nn(train_X.T, trY, test_X.T, teY, itr, dims, wts, lc=4)
    print("Encoder and Softmax combined layer:  Test data loss = ", vals[0]," accuracy = ", vals[1]*100)

    
    
if __name__ == "__main__":
    main()