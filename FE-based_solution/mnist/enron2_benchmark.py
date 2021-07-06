"""
Runs encryption and decryption on the entire MNIST test set, and reports
timings.
"""
import setup
import os
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '0'
    os.execv(sys.executable, [sys.executable] + sys.argv)

from charm.toolbox.pairinggroup import pair
from core import (
    discretelogarithm,
    models,
    scheme,
)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.models as Km
import keras.layers as Kl

# from sklearn.datasets import fetch_mldata
import timeit
import time
import datetime
# assert False, 'Remember to run with -O for optimized results.'

from datetime import date
today = date.today()
d = today.strftime("%b-%d-%Y")

file_name = "entire_enron2_email_FE_ML_output_%s.txt" %d
print("++++++++++++++++++++++++++++++++++++++++++++++\n", file=open(file_name, "a"))
print("++++++++++++++++++++++++++++++++++++++++++++++\n", file=open(file_name, "a"))
time2 = timeit.default_timer() # start the test
# ff = open(file_name, "w")
# print("Start printing logs!", file=open(file_name, "a")

inst = 'objects/instantiations/MNT159.inst'
vector_length = 5000 #33510 #2000 # for news; 2000 for emails
# print("Vector length =  %s" %str(vector_length), file=open(file_name, "a"))
k = 40
classes = 20
model ='objects/ml_models/enron2_SpamModel_{}-40-20-10-2.mlm'.format(vector_length)
#X, y = mnist["data"].astype('float'), mnist["target"].astype('float')

X_test = np.load('enron2/enron2_email_X_test_vector_length_5000_1.npy').astype('float')
y_test = np.load('enron2/enron2_email_dummy_y_test_X_5000_1.npy').astype('float')


# print("length of X_test: " + str(len(X_test)))
# print("length of y_test: " + str(len(y_test)))
# Randomly shuffle test set (with a set seed)
# np.random.seed(42)
# sigma = np.random.permutation(len(X_test))
# X_test = X_test[sigma]
# y_test = y_test[sigma]

print('Importing scheme.')

scheme = scheme.ML_DGP(inst)
print('Done!\n')

print('Importing model.')
ml = models.MLModel(source=model)

biased = np.ones(vector_length+1)

print('Done!\n')



print('Importing discrete logarithm.')

dlog = discretelogarithm.PreCompBabyStepGiantStep(
    scheme.group,
    scheme.gt,
    minimum=-1.7e+11,
    maximum=2.7e+11,
    step=1 << 13,
)

scheme.set_dlog(
    dlog
)
print('Done!\n')

print('Importing keys...')
pk = models.PublicKey(
    source='objects/pk/common_{}.pk'.format(vector_length)
)
msk = models.MasterKey(
    source='objects/msk/common_{}.msk'.format(vector_length)
)
print('Done!\n')



print('Generating functional decryption key...')
dk = scheme.keygen(msk, ml) # using model parameters to generate a functional decryption key
print('Done!\n')

#===========================================#
enc_total = 0
eval_total = 0
dlog_total = 0
dec_total = 0
dectime = 0
enctime = 0
predict_time = 0

correct = 0
errors = 0


# pre-trained whole model
# model = keras.Sequential([
#     keras.layers.InputLayer(input_shape=(first_layer_parameter,), name='input_layer'),
#     keras.layers.Dense(40, name='hidden_1'),
#     keras.layers.Lambda(lambda x: x * x, name='lambda_layer'),
#     keras.layers.Dense(20, name='hidden_2'), #output is vector of 20
#     keras.layers.Dense(10, name='hidden_3'),
#     keras.layers.Dense(2,activation='sigmoid', name='output_layer')
# ])

## unencrypted part of the model
modelPub = keras.Sequential([
    keras.layers.InputLayer(input_shape=(20,)),
    keras.layers.Dense(10, name = 'hidden_3', input_shape=(20,)),
    keras.layers.Dense(2, activation='sigmoid',name='output')
])

dir = "enron2"
HiddenLayer3_Weight = dir + "/sequential_hidden_3_MatMul_ReadVariableOp_transpose.npy"
HiddenLayer3_Bias = dir + "/sequential_hidden_3_MatMul_bias.npy"

Output_layer_Weight = dir + "/sequential_output_layer_MatMul_ReadVariableOp_transpose.npy"
Ouput_layer_Bias = dir + "/sequential_output_layer_MatMul_bias.npy"



QuanHiddenLayer3_Weight = np.load(HiddenLayer3_Weight)
# print('len of QuanHiddenLayer3_Weight [1]= ', str(len(QuanHiddenLayer3_Weight[1])))
# print('QuanHiddenLayer3_Weight= \n{}' .format(QuanHiddenLayer3_Weight))
# print('size of QuanHiddenLayer3_Weight', str(size(QuanHiddenLayer3_Weight)))
QuanHiddenLayer3_Bias = np.load(HiddenLayer3_Bias)
# print('QuanHiddenLayer3_Bias= \n{}' .format(QuanHiddenLayer3_Bias))
# print('len of QuanHiddenLayer3_Bias[1] = ', str(len(QuanHiddenLayer3_Bias[1])))
QuanOutput_layer_Weight = np.load(Output_layer_Weight)
# print('len of QuanOutput_layer_Weight[1] = ', str(len(QuanOutput_layer_Weight[1])))
# print('QuanOutput_layer_Weight= \n{}' .format(QuanOutput_layer_Weight))
QuanOuput_layer_Bias = np.load(Ouput_layer_Bias)
# print('len of QuanOuput_layer_Bias[1] = ', str(len(QuanOuput_layer_Bias[1])))
# print('QuanOuput_layer_Bias= \n{}' .format(QuanOuput_layer_Bias))

ListHiddenLayer_3 = []
ListOutputLayer = []
ListHiddenLayer_3.append(QuanHiddenLayer3_Weight.transpose())
ListHiddenLayer_3.append(QuanHiddenLayer3_Bias.transpose())
# print('ListHiddenLayer_3 = {}' .format(ListHiddenLayer_3))
ListOutputLayer.append(QuanOutput_layer_Weight.transpose())
ListOutputLayer.append(QuanOuput_layer_Bias.transpose())

# print('shape of ListHiddenLayer_3 = ', str(ListHiddenLayer_3[1].shape))
# print('shape of ListOutputLayer = ', str(ListOutputLayer[1].shape))

modelPub.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
modelPub.layers[0].set_weights(ListHiddenLayer_3)
modelPub.layers[1].set_weights(ListOutputLayer)

# print('ListHiddenLayer_3 = {}' .format(ListHiddenLayer_3))
# print('ListOutputLayer = {}' .format(ListOutputLayer))
#
# print('modelPub weights = {}'.format(modelPub.get_weights()))
# sys.exit()


# num_test = len(X_test)
num_test = 5
# print("Number of emails tested = %s" % (str(num_test)))

for i in range(num_test):
    print('i = ', str(i))
    biased[1:] = X_test
    # print('X_test [i] =')
    # print(X_test[i])
    # print('biased =')
    # print(biased)
    results = ml.evaluate(biased) # classify on plaintext data
    # print('Output of classifying on plaintext data after hidden 2:\n{}'.format(results))

    v = models.Vector(array=X_test) # put X into a vector

    # encryption
    before_encrypt = timeit.default_timer()
    c = scheme.encrypt(pk, v)                # encrypting data
    after_encrypt = timeit.default_timer()
    enc_time = after_encrypt - before_encrypt
    enc_total += enc_time

    # encryption_time = time2 - time1
    # enctime += encryption_time

    time1 = timeit.default_timer()  # start prediction
    # decryption
    before_dec = timeit.default_timer()
    dec = scheme.decrypt(pk, dk, c)
    after_dec = timeit.default_timer()

    # decryption_time = time2 - time1
    # dlog_time = after_dlog - before_dlog
    # dlog_total += dlog_time


    # dectime += decryption_time

    # print('Ouput on encrypted data after hidden 2:\n{}'.format(dec))
    # # print('size of dec = {}'.format(size(dec)))
    # print('dec =')
    # print(dec)


    # need to carefu about the broadcast error when deal with array/list dimension
    dec = np.reshape(np.asarray(dec),(1,20))
    # print(np.asarray(dec).shape)
    # modelPub.summary()
    pred_label = modelPub.predict(dec) # predicted label
    time2 = timeit.default_timer() # end prediction
    predict_time = time2 - time1
    dec_time = after_dec - before_dec
    dec_total += dec_time

    # print("res=")
    # print(res)
    # print("y_label =")
    # print(y_test[i])

    if (np.argmax(y_test) == np.argmax(pred_label)):
        correct +=1
    else:
        errors +=1
        print("error!!!!")

time2 = timeit.default_timer() #finished the test
exprm_time = time2 - time1

print('Running stats:', file=open(file_name, "a"))

print('=================', file=open(file_name, "a"))
print('Average accuracy: {}.'.format(100 * correct / (num_test)), file=open(file_name, "a"))
print('Total errors: {}.'.format(errors), file=open(file_name, "a"))
print('=================', file=open(file_name, "a"))
print('Average encryption time: {}s.'.format(enc_total/(num_test)), file=open(file_name, "a"))
print('Average decryption time: {}s.'.format(dec_total / (num_test)), file=open(file_name, "a"))

print('=================', file=open(file_name, "a"))
print("Runtime for one encryption: {}." .format(enc_time), file=open(file_name, "a"))
print("Runtime for one decryption: {}." .format(dec_time), file=open(file_name, "a"))
print("Runtime for one prediction: {}." .format(predict_time), file=open(file_name, "a"))
print("Runtime for exprm_time {}." .format(exprm_time), file=open(file_name, "a"))
