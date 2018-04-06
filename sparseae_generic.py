# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:49:30 2018

@author: nhjadhav
"""
from keras.datasets import mnist
from load_dataset import mnist
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
    #weights = tf.random_normal(shape)
    xinit = tf.contrib.layers.xavier_initializer()
    return tf.Variable(xinit(shape))


def image_plots(img,name,p):
    figure, axes = plt.subplots(nrows=10, ncols=10)
    index = 0
    for axis in axes.flat:
        """ Add row of weights as an image to the plot """
        image = axis.imshow(img[index, :].reshape(28, 28),cmap=plt.cm.gray, interpolation='nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1

    """ Show the obtained plot """
    file=name+"_"+str(p)+".png"
    plt.savefig(file)
    plt.close() 


def se_autoencoder(data,p):
    
    print("Model generating for p = ", p)
    #=============initialize the weights======================
    n_in=784
    n_out=200
    W1=init_weights((n_out,n_in))
    b1=init_weights((n_out,1))
    
    #Swap for W2 here
    W2=init_weights((n_in,n_out))
    b2=init_weights((n_in,1))
    
    activation=tf.nn.sigmoid
    optimizer=tf.train.GradientDescentOptimizer(0.1)
    #optimizer=tf.train.AdamOptimizer()
    lval=0.001
    beta=3
    X = tf.cast(data, tf.float32)
    
    #=============Generate the model======================  
    A1 = activation(tf.matmul(W1,X) + b1)
    X_hat = activation(tf.matmul(W2,A1) + b2)
    p_hat=tf.reduce_mean(A1,axis=0)
    
    
    kld=get_kld(p, p_hat)
    #cost=tf.losses.mean_squared_error(X,X_hat)  + 0.5*lval*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) + beta*tf.reduce_sum(kld)
    diff = X - X_hat
    cost = tf.reduce_mean(tf.reduce_sum(diff**2,axis=0)) + 0.5*lval*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)) + beta*tf.reduce_mean(kld)
    weight_list=[W1,W2]
    train_step=optimizer.minimize(cost, var_list=weight_list)
    
    #========================train the model================================================
    init = tf.global_variables_initializer()    
    sess.run(init)  
    
    for i in range(400):
        sess.run(train_step)
        if(i%100 == 0):
            print("cost at ", i, cost.eval(session=sess))
           
    weight_img=W1.eval(session=sess)
    original_img=X.eval(session=sess)
    reconstructed_img=X_hat.eval(session=sess)
    
    #========================Plot the weight and reconstruced images============================================
    
    image_plots(weight_img,"W1",p)
    image_plots(original_img.T,"X",p)
    image_plots(reconstructed_img.T,"X_hat",p)
    
    print("Done for p = ", p)
    
def main():
    train_data, train_label, test_data, test_label =  mnist(ntrain=6000,ntest=1000,digit_range=[0,10])
    print("X shape is ", train_data.shape)

    p_vals= [0.01, 0.1, 0.5, 0.8]
    for p in p_vals:
        se_autoencoder(train_data,p)
    
if __name__ == "__main__":
    main()