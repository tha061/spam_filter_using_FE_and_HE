from sklearn.model_selection import train_test_split
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pathlib
from keras.utils import np_utils
from keras import backend as K
import keras.models as Km
import keras.layers as Kl
from tensorflow import keras
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from keras.losses import categorical_crossentropy
from sklearn.feature_selection import mutual_info_classif
from contextlib import redirect_stdout



tf.compat.v1.enable_eager_execution()



from datetime import date
today = date.today()
d = today.strftime("%b-%d-%Y")
file_name = "email_Prediction_output_%s_collateral.txt" %d
print("++++++++++++++++++++++++++++++++++++++++++++++\n", file=open(file_name, "a"))
print("++++++++++++++++++++++++++++++++++++++++++++++\n", file=open(file_name, "a"))
# print("Start printing logs!", file=open(file_name, "a")

# Note: the email length is length than 100, the word is occur than 2 times
lableDic = {}
with open('categoryIndex') as f:
    for line in f:
        line = line.strip()
        lable, key = line.split(';')
        lableDic[key] = lable


LABELS_FILE_OUTPUT = 'sourceIndex'
#----------------------------------------------------------------------
# split the dataset from formated email file,
# input: the index_file contain the result and original email path
# Output: train_dataset, train_result, test_dataset, test_result
#-----------------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import nltk




LABELS_FILE_OUTPUT = 'sourceIndex'


lableDic = {}
with open('categoryIndex') as f:
    for line in f:
        line = line.strip()
        lable, key = line.split(';')
        lableDic[key] = lable

def split_data_set(Index_file):
    total = int(os.popen("cat " + Index_file + "|wc -l").read())
    print("total is " + str(total))
    X = [None] * (total)
    y = [None] * (total) # public label
    
    
    # private label
    XPresent = []
    XNotPresent = []
   # XTech = []
    
    # main label
    XSpam = []
    XHam = []
    
    SignalPresent = 0
    SignalNotPresent = 0
    #SignalTech = 0
    SignalSpam = 0
    SignalHam = 0
    
    yPrivate = [None] * (total) # private label
    counter = 0
    counter1 = 0
    max_email = 0
    with open(Index_file) as f:
        for line in f:
            if counter % 10000 == 0:
                print("Input Processing in Progress .......... ", str(counter))

            line = line.strip()
            label, key = line.split(';')
            
            # private label
            if lableDic[key] == 'present':
                yPrivate[counter] = 1
                SignalPresent = 1
            elif lableDic[key] == 'notpresent':
                yPrivate[counter] = 0
                SignalNotPresent = 1
          #  elif lableDic[key] == 'technology':
           #     yPrivate[counter] = 0
            #    SignalTech = 1
            
            # public label
            if label.lower() == 'spam':
                y[counter] = 1
                SignalSpam = 1
            elif label.lower() == 'ham':
                y[counter] = 0
                SignalHam = 1
            counter += 1


            fileName = key
            newfileName = "catemail_out/" + fileName
            with open(newfileName, 'r') as outputF:
                X[counter1] = outputF.read()
#            if ((len(X[counter].split()) > 10000) or (len(X[counter].split()) < 10)) or (len(X[counter].split()) == 0):
#                counter = counter -1
#                max_email += 1
                counter1 = counter1 + 1
            
            with open(newfileName, 'r') as outputF:
                content = outputF.read()
                if(SignalPresent == 1):
                    XPresent.append(content)
                    SignalPresent = 0
                elif SignalNotPresent ==1:
                    XNotPresent.append(content)
                    SignalNotPresent = 0
                #elif SignalTech ==1:
                #    XTech.append(content)
                #    SignalTech = 0
                if(SignalSpam ==1):
                    XSpam.append(content)
                    SignalSpam = 0
                elif(SignalHam == 1):
                    XHam.append(content)
                    SignalHam = 0
    print("the number of news are deleted : " + str(max_email))
    print("the number of total news: " + str(counter1))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
    X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(X, yPrivate, test_size=0.30, random_state=2)
    return X_train, X_test,y_train, y_test, X_train_pri, X_test_pri, y_train_pri, y_test_pri, XPresent, XNotPresent, XSpam, XHam


#---------------------------------------------------------------------
# split the dataset into train_data and test_data
# result
#   X_train
#   X_test
#   y_train
#   y_test
#---------------------------------------------------------------------
X_train, X_test, y_train, y_test, X_train_pri, X_test_pri, y_train_pri, y_test_pri,XPresent, XNotPresent, XSpam, XHam = split_data_set(LABELS_FILE_OUTPUT)


#---------------------------------------------------------------------
# create the document-term-matrix
# result
#   X_train_vector
#   X_test_vector
#---------------------------------------------------------------------
vectorizerPresent = CountVectorizer(max_features = 100)
vectorizerNotPresent = CountVectorizer(max_features = 100)
#vectorizerTech = CountVectorizer(max_features = 100)
vectorizerSpam = CountVectorizer(max_features = 100)
vectorizerHam = CountVectorizer(max_features = 100)

Trainvectorizer = CountVectorizer()
X_train_vector = Trainvectorizer.fit_transform(X_train)

XPresentVector = vectorizerPresent.fit_transform(XPresent)
XNotPresentVector = vectorizerNotPresent.fit_transform(XNotPresent)
#XTechVector = vectorizerTech.fit_transform(XTech)
XSpamVector = vectorizerSpam.fit_transform(XSpam)
XHam = vectorizerHam.fit_transform(XHam)

TrainVocabulary = Trainvectorizer.get_feature_names()
PresentVocabulary = vectorizerPresent.get_feature_names()
NotPresentVocabulary = vectorizerNotPresent.get_feature_names()
#TechVocabulary = vectorizerTech.get_feature_names()
SpamVocabulary = vectorizerSpam.get_feature_names()
HamVocabulary = vectorizerHam.get_feature_names()


UnionVocabulary = PresentVocabulary + NotPresentVocabulary + SpamVocabulary + HamVocabulary
UnionUniqVocabulary = list(set(UnionVocabulary)) # get all unique vocabulary
print("all unique vocab length = ")
print(len(UnionUniqVocabulary))

count = 0
for item in UnionUniqVocabulary:
    if item not in TrainVocabulary:
        print(item)
    else:
        count += 1
    if count == len(UnionUniqVocabulary):
        print("all included")

	
vec = 2000
first_layer_parameter = vec # we choose the top 2000 words


# public task
res = dict(zip(Trainvectorizer.get_feature_names(),mutual_info_classif(X_train_vector, y_train, discrete_features=True)))
a = {k: v for k, v in sorted(res.items(), key=lambda item: item[1],reverse=True)}
print("no of vocab in training set regard publ label= ")
print(len(a)) #number of vocab
listMIPub = list(a.keys())[:vec] # feature vector size with pub label, can choose top 2000, 3000 ...

# private task
res = dict(zip(Trainvectorizer.get_feature_names(),mutual_info_classif(X_train_vector, y_train_pri, discrete_features=True)))
a = {k: v for k, v in sorted(res.items(), key=lambda item: item[1],reverse=True)}
print("no of vocab in training set regard private label= ")
print(len(a))
listMIPri = list(a.keys())[:vec] # feature vector size with pub label, can choose top 2000, 3000 ...

# first_layer_parameter = len(a)

def reform_data_set(Index_file):
    total = int(os.popen("cat " + Index_file + "|wc -l").read())
    print("total is " + str(total))
    X = [None] * (total)
    y = [None] * (total)
    XPresent = []
    XNotPresent = []
    #XTech = []
    XSpam = []
    XHam = []
    SignalPresent = 0
    SignalNotPresent = 0
    #SignalTech = 0
    SignalPresent = 0
    SignalNotPresent = 0
    yPrivate = [None] * (total)
    counter = 0
    deleted_email = 0
    with open(Index_file) as f:
        for line in f:
            if counter % 10000 == 0:
                print("Reform dataset: Input Processing in Progress .......... ", str(counter))
            line = line.strip()
            label, key = line.split(';')
            # private task
            if lableDic[key] == 'present':
                yPrivate[counter] = 1
                SignalPresent = 1
            elif lableDic[key] == 'notpresent':
                yPrivate[counter] = 0
                SignalNotPresent = 1
            
                
            # public task
            if label.lower() == 'spam':
                y[counter] = 1
                SignalSpam = 1
            elif label.lower() == 'ham':
                y[counter] = 0
                SignalHam = 1

            fileName = key
            newfileName = "catemail_out/" + fileName
            X[counter] = load(newfileName)
            counter = counter + 1
    print("reform data: the number of mails are deleted : " + str(deleted_email))
    print("reform data: the number of total email: " + str(counter))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
    X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(X, yPrivate, test_size=0.30, random_state=2)
    return  X_train, X_test,y_train, y_test, X_train_pri, X_test_pri, y_train_pri, y_test_pri


def load(path):
    email_text = ""
    with open(path, 'r') as test_file:
        email_text = test_file.read()

    # Tokenize the message
    tokens = nltk.word_tokenize(email_text)

    # Remove the tokens that are greater than 15 characters
    tokens = [x for x in tokens if (x in listMIPub) # or (x in listMIPri))]

    # Combine the tokens with space and build a single string again.
    return " ".join([i.strip() for i in tokens])


X_train, X_test, y_train, y_test, X_train_pri, X_test_pri, y_train_pri, y_test_pri = reform_data_set(LABELS_FILE_OUTPUT)

vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)


print("The length of X_train_vector: %s" %(str(X_train_vector.shape)), file=open(file_name, "a"))
print("The length of X_test_vector: %s" %(str(X_test_vector.shape)), file=open(file_name, "a"))


#---------------------------------------------------------------------
# convert the document-term-matrix to 0 - 255
# result
#   X_train_vector
#   X_test_vector
#---------------------------------------------------------------------
maxFrequenceT = np.amax(X_train_vector)
maxFrequenceT1 = np.amax(X_test_vector)
if (maxFrequenceT > maxFrequenceT1):
    maxFrequence = maxFrequenceT
else:
    maxFrequence = maxFrequenceT1
X_train_vector = np.around((X_train_vector / maxFrequence ) * 255)
X_test_vector = np.around((X_test_vector / maxFrequence ) * 255)
np.save('news_X_test_vector_1', X_test_vector.toarray()[1])
np.save('news_X_test_vector_entire', X_test_vector.toarray())





#---------------------------------------------------------------------
# transfer the result to 2 type
# result
#   dummy_y_train
#   dummy_y_test
#   dummy_y_train_private
#   dummy_y_test_private
#---------------------------------------------------------------------
encode = LabelEncoder()
encode.fit(y_train)
encoded_Y = encode.transform(y_train)
dummy_y_train = np_utils.to_categorical(encoded_Y)
# print(y_train)
encode2 = LabelEncoder()
encode2.fit(y_test)
encoded_Y2 = encode2.transform(y_test)
dummy_y_test = np_utils.to_categorical(encoded_Y2)
np.save('emails_dummy_y_test_1', dummy_y_test[1])
# np.save('emails_dummy_y_test_entire', dummy_y_test)
# print(dummy_y_train.shape)



## private task
encode3 = LabelEncoder()
encode3.fit(y_train_pri)
encoded_Y3 = encode3.transform(y_train_pri)
dummy_y_train_private = np_utils.to_categorical(encoded_Y3)
encode4 = LabelEncoder()
encode4.fit(y_test_pri)
encoded_Y4 = encode4.transform(y_test_pri)
dummy_y_test_private = np_utils.to_categorical(encoded_Y4)




#---------------------------------------------------------------------
# create the ML model, linear -> X*X -> linear
# create factor to get the output
# result
#   model
#---------------------------------------------------------------------
### Model for main task: spam classification
first_layer_parameter = X_train_vector.shape[1]
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(first_layer_parameter,)),
    keras.layers.Dense(40),
    keras.layers.Lambda(lambda x: x * x),
    keras.layers.Dense(20),
    keras.layers.Dense(10),
    keras.layers.Dense(2,activation='sigmoid')
])
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy']);

# print("-----------------------------------------------------------", file=open(file_name, "a"))
# print("-----------------the Q task with 2 output------------------", file=open(file_name, "a"))
#print(model.summary(), file=fi)
with open(file_name, 'a') as ff:
	print('Before_Quant_modelsummary:', file=ff)
	with redirect_stdout(ff):
		model.summary()


print("X_train_vector.shape: ")
print(X_train_vector.shape)
print("dummy_y_train shape")
print(dummy_y_train.shape)
# training phase
model.fit(X_train_vector.toarray(), dummy_y_train, epochs=30, verbose=2);
# evaluate on test data
loss, accuracy = model.evaluate(X_test_vector.toarray(), dummy_y_test, verbose=2);
print('Main task: Accuracy before quant: %f' % (accuracy*100), file=open(file_name, "a"))

# prediction
modelPredictionTrain = model.predict(X_train_vector.toarray())
modelPredictionTest = model.predict(X_test_vector.toarray())



#--------------------------------------------------------------------
# convert the model weight to integer
# result
#   mail_model.tflite  -- file contains lite model
#   mail_model_quant.tflite -- file contains int model
#   mail_model_quant_io.tflite -- file contains only int model
#--------------------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_models_dir = pathlib.Path(".")
mail = tf.cast(X_train_vector.toarray(), tf.float32)
mail_ds = tf.data.Dataset.from_tensor_slices((mail)).batch(1)
def representative_data_gen():
  for input_value in mail_ds.take(33125):
    yield [input_value]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()
trained_model_name = "emails_mainTask_model_quant_io.tflite" 
tflite_model_quant_file = tflite_models_dir/trained_model_name
# tflite_model_quant_file = tflite_models_dir/"news_model_quant_io.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)


#---------------------------------------------------------------------------------
# test the new quant model
#---------------------------------------------------------------------------------

tflite_model_quant_file = tflite_models_dir/"emails_mainTask_model_quant_io.tflite"
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter_quant.allocate_tensors()
# input_index_quant = interpreter_quant.get_input_details()[0]["index"]
# output_index_quant = interpreter_quant.get_output_details()[0]["index"]
# test_mail = np.expand_dims(X_test_vector.toarray()[1], axis=0).astype(np.float32)
# interpreter_quant.set_tensor(input_index_quant, test_mail)
# interpreter_quant.invoke()
# predictions = interpreter_quant.get_tensor(output_index_quant)


#----------------------------------------------------------------------------------
# evaluate the models for main task
# result:
#   predictResult
#----------------------------------------------------------------------------------
def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    prediction_digits = []
    for test_data in X_test_vector.toarray():
        test_data = np.expand_dims(test_data, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_data)
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)
    accurate_count = 0
    error = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == y_test[index]:
            accurate_count += 1
        else:
            error += 1
    print("total error = %s " % str(error))
    accuracy = accurate_count * 1.0 / len(prediction_digits)
    return accuracy * 100
X_test_vector.toarray()
print("Main Task: Accuracy after quant: %s" % (str(evaluate_model(interpreter_quant))), file=open(file_name, "a"))

#### NOTES
'''
Here the code will be terminated for the first time running.
The reason is we want to get the weights and bias of the quantized model "news_model_quant_io_Nov-20-2020.tflite"
We can use netron app to get the layer's weights and bias and use them later in the code:
https://netron.app/

We will recived npy files for all weights and bias of all layers using this app: something like below (ote the name might be different)
sequential_output_layer_MatMul_ReadVariableOp_transpose.npy"
sequential_hidden_1_MatMul_bias.npy"
sequential_hidden_1_MatMul_ReadVariableOp_transpose.npy"
sequential_hidden_2_MatMul_bias.npy"
sequential_hidden_2_MatMul_ReadVariableOp_transpose.npy"
sequential_hidden_3_MatMul_bias.npy"
sequential_hidden_3_MatMul_ReadVariableOp_transpose.npy"
sequential_output_layer_MatMul_bias.npy"
'''
####

sys.exit()

##### Notes
'''
Commented out the sys.exit() and run the python code again. This time, we ignore the new .tflite file
Just need to specify the directory of .npy files from the first two layers (ignore the lamda layer) in the below code (lines 531-534)
'''


