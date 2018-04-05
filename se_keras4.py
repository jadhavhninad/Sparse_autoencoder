from keras import backend as K
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


sp = 0.01
b_val = 3;#Controls the acitvity of the hidden layer nodes
encoding_dim = 200  
input_img = Input(shape=(784,))
lambda_val = 0.001 #Weight decay , refer : https://stackoverflow.com/questions/44495698/keras-difference-between-kernel-and-activity-regularizers

#Custom Regularizer function
def sparse_reg(activ_matrix):
    p = 0.01
    beta = 3
    p_hat = K.mean(activ_matrix) # average over the batch samples
    print("p_hat = ",p_hat)
    #KLD = p*(K.log(p)-K.log(p_hat)) + (1-p)*(K.log(1-p)-K.log(1-p_hat))
    KLD = p*(K.log(p/p_hat)) + (1-p)*(K.log(1-p/1-p_hat))
    print("KLD = ", KLD)
    return beta * K.sum(KLD) # sum over the layer units


encoded = Dense(encoding_dim, 
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(lambda_val/2),activity_regularizer=sparse_reg)(input_img)

decoded = Dense(784,
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(lambda_val/2),activity_regularizer=sparse_reg)(encoded) #Switch to softmax here?

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='sgd', loss='mse') #What optimizer to use??

#For predicting the output of the encoded and decoded layers
encoder = Model(input_img, encoded) #map input image to decoded image

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1] #Gets the last layer
decoder = Model(encoded_input, decoder_layer(encoded_input))


#Get the train and test as done in DL assignments.
(x_train, _), (x_test, _) = mnist.load_data()

#Removing normalization since using MSE - later
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
                       
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


autoencoder.fit(x_train, x_train,
                epochs=400,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

print(autoencoder.summary())
'''
======================CLASSIFICATION OF TEST DATA=========================================================================
'''
# encode and decode some digits
# note that we take them from the *test* set. This is testing that the digits are correctly classified.
#Why this is requried? Why not send the images directly and the encoding happens internally in the model and we get the output? Explore.
encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)



n = 1  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
