# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:49:30 2018

@author: nhjadhav
"""
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.Session()
    
def get_kld(p, p_hat):
    kl_val =  p * tf.log(p) - p * tf.log(p_hat) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_hat)
    #weight_img=kl_val.eval(session=sess)
    #print("KLD is ", weight_img.shape)
    #print(kl_val)
    return kl_val

def init_weights(shape):
    weights = tf.random_normal(shape) * 0.01
    return tf.Variable(weights)

def se_autoencoder(data,p):
    
    
    #initialize the weights
    n_in=784
    n_out=200
    W1=init_weights((n_out,n_in))
    b1=init_weights((n_out,1))
    
    #Swap for W2 here
    W2=init_weights((n_in,n_out))
    b2=init_weights((n_in,1))
    
    activation=tf.nn.sigmoid
    #optimizer=tf.train.GradientDescentOptimizer(0.01)
    optimizer=tf.train.AdamOptimizer()
    lval=0.001
    beta=3
    X=tf.placeholder("float",shape=[None,data.shape[1]])
    X=data
    
    '''
    print("======before starting X is======")
    weight_img=X.eval(session=sess)
    print(weight_img[:,0])
    '''

    '''
    weight_img=W2.eval(session=sess)
    print(weight_img[0,0:10])
    '''
    
    #for i in range(400):
        
    A1 = activation(tf.matmul(W1,X) + b1)
    X_hat = activation(tf.matmul(W2,A1) + b2)
    p_hat=tf.reduce_mean(A1,axis=0)
    
    '''
    weight_img=X.eval(session=sess)
    print("X is ", weight_img.shape)
    weight_img=A1.eval(session=sess)
    print("A1 is ", weight_img.shape)
    weight_img=X_hat.eval(session=sess)
    print("X_hat is ", weight_img.shape)
    weight_img=p_hat.eval(session=sess)
    print("p_hat is ",weight_img.shape)
    weight_img=W1.eval(session=sess)
    print("W1 is ",weight_img.shape)
    weight_img=W2.eval(session=sess)
    print("W2 is ",weight_img.shape)
    weight_img=b1.eval(session=sess)
    print("b1 is ",weight_img.shape)
    weight_img=b2.eval(session=sess)
    print("b2 is ",weight_img.shape)
    '''
    
    kld=get_kld(p, p_hat)
    #cost=tf.losses.mean_squared_error(X,X_hat)  + 0.5*lval*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) + beta*tf.reduce_sum(kld)

    diff = X - X_hat
    cost = 0.5*tf.reduce_mean(tf.reduce_sum(diff**2,axis=0)) + 0.5*lval*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) + beta*tf.reduce_sum(kld)
    weight_list=[W1,W2]
    train_step=optimizer.minimize(cost, var_list=weight_list)
    
    '''
    print("----------Checking the value of X and Xhat----------")
    weight_img=X.eval(session=sess)
    print(weight_img[:,0])
    print(weight_img.shape)
    weight_img=X_hat.eval(session=sess)
    print(weight_img[:,0])
    
    print("------During training, W is----------")
    weight_img=W1.eval(session=sess)
    print(weight_img[0,0:10])
    weight_img=W2.eval(session=sess)
    print(weight_img[0,0:10])
    '''
    
    init = tf.global_variables_initializer()    
    sess.run(init)
    
        
    print("=======W1 before : ================")
    weight_img=W1.eval(session=sess)
    print(weight_img[0,0:10])
    
    
    for i in range(400):
        sess.run(train_step)
    '''
    for i in range(10):
        sess.run(train_step)
        print("During training, W is")
        weight_img=W1.eval(session=sess)
        print(weight_img[0,0:10])
    '''
    
    weight_img=W1.eval(session=sess)
    print("W1 After")
    print(weight_img[0,0:10])
    
    figure, axes = plt.subplots(nrows=10, ncols=10)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """
        image = axis.imshow(weight_img[index, :].reshape(28, 28),cmap=plt.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """
    file="se_ae_400_"+str(p)+".png"
    plt.savefig(file)
    plt.close()
    print("Done for ",p)
    

    original_img=X.eval(session=sess).T
    
    figure, axes = plt.subplots(nrows=10, ncols=10)
    index = 0

    for axis in axes.flat:
        """ Add row of weights as an image to the plot """
        image = axis.imshow(original_img[index, :].reshape(28, 28),cmap=plt.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """
    file="se_ae_original_"+str(p)+".png"
    plt.savefig(file)
    plt.close() 
    
def main():
    #Get the train and test data. No labels.
    (x_train, _), (x_test, _) = mnist.load_data()
    
    #Removing normalization since using MSE - later
                           
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    X = np.concatenate((x_train, x_test), axis=0).T
    #X = X[:,0:1000]
    print("X shape is ", X.shape)
       
    X = tf.cast(X, tf.float32)
    
    p_vals=[0.01]
    for p in p_vals:
        se_autoencoder(X,p)
    
if __name__ == "__main__":
    main()