
import os
import sys
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
	os.environ['PYTHONHASHSEED'] = '0'
	os.execv(sys.executable, [sys.executable] + sys.argv)
	
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.models as Km
import keras.layers as Kl
import numpy as np
import pickle

import setup

from core import (
    discretelogarithm,
    make_keys,
    models,
    scheme
)

from datetime import date
today = date.today()
d = today.strftime("%b-%d-%Y")

inputLength = 5000 #news: 24589; inputLength=2000 #emails
#ModelName = "email_SpamModel_" + today.strftime("%b-%d-%Y") + str(inputLength)
ModelName = "ceas_SpamModel_{}-40-20-10-2".format(inputLength) # + today.strftime("%b-%d-%Y")

#FirstLayerWeight = "/home/ubuntu/" + str(inputLength) + "/" + str(inputLength) + "_40_weight_layer1.npy"
#FirstLayerBias = "/home/ubuntu/" + str(inputLength) + "/" + str(inputLength) + "_40_bias_layer1.npy"
#SecondLayerWeight = "/home/ubuntu/" + str(inputLength) + "/40_2_weight_layer2.npy"
#SecondLayerBias = "/home/ubuntu/" + str(inputLength) + "/40_2_bias_layer2.npy"
dir = "ceas08_5000"
FirstLayerWeight = dir + "/sequential_hidden_1_MatMul_ReadVariableOp_transpose.npy"
FirstLayerBias = dir+ "/sequential_hidden_1_MatMul_bias.npy"
SecondLayerWeight = dir + "/sequential_hidden_2_MatMul_ReadVariableOp_transpose.npy"
SecondLayerBias = dir+ "/sequential_hidden_2_MatMul_bias.npy"
#sequential_dense_1_bias.npy
dense_1_weight = np.load(FirstLayerWeight)
dense_1_bias = np.load(FirstLayerBias)
dense_2_weight = np.load(SecondLayerWeight)
dense_2_bias = np.load(SecondLayerBias)
print("dense_1_weight is: " + str(dense_1_weight.shape))
print("dense_1_bias is: " + str(dense_1_bias.shape))
print("dense_2_weight is: " + str(dense_2_weight.shape))
print("dense_2_bias is: " + str(dense_2_bias.shape))

tempProj = np.ones((dense_1_weight.shape[0],dense_1_weight.shape[1]+1))
tempProj[:,0] = dense_1_bias
tempProj[:,1:] = dense_1_weight
newTempProj = np.transpose(tempProj)
New_dense_2_weight = np.transpose(dense_2_weight)

SpamProject = models.Projection(matrix=newTempProj)
SpamForms = models.DiagonalQuadraticForms(matrix=New_dense_2_weight)
SpamModel = models.MLModel(proj = SpamProject, forms = SpamForms)
SpamModel.toFile('objects/ml_models', ModelName)

# ThirdLayerWeight = "/home/ubuntu/tham_project/corey/project/reading-in-the-dark/mnist/email_trained-2000-40-20-10-2/sequential_hidden_3_MatMul_ReadVariableOp_transpose.npy"
# ThirdLayerBias = "/home/ubuntu/tham_project/corey/project/reading-in-the-dark/mnist/email_trained-2000-40-20-10-2/sequential_hidden_3_MatMul_bias.npy"
# OutputLayerWeight = "/home/ubuntu/tham_project/corey/project/reading-in-the-dark/mnist/email_trained-2000-40-20-10-2/sequential_output_layer_MatMul_ReadVariableOp_transpose.npy"
# OutputLayerBias = "/home/ubuntu/tham_project/corey/project/reading-in-the-dark/mnist/email_trained-2000-40-20-10-2/sequential_output_layer_MatMul_bias.npy"

