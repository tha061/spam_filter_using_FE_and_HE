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

# first_layer_parameter = 30000
amin_df = 900
LABELS_FILE_OUTPUT = 'trec07p/full/index'

from datetime import date
from datetime import datetime
today = date.today()
d1 = today.strftime("%b-%d-%Y")
now = datetime.now()

# dd/mm/YY H:M:S
d = now.strftime("%d-%m-%Y-%H:%M:%S")
# print("date and time =", dt_string)

vec = 500

file_name = "trec07-results/email_Prediction_output_{0}_40_20_10_2_{1}.txt".format(vec, d1)
print("++++++++++++++++++++++++++++++++++++++++++++++\n", file=open(file_name, "a"))
print("++++++++++++++++++++++++++++++++++++++++++++++\n", file=open(file_name, "a"))
print("now = %s" %str(d), file=open(file_name, "a"))
# fi = open(file_name, "a")
# print("Start printing logs!", file=open(file_name, "a")

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


def split_data_set(Index_file):
    total = int(os.popen("cat " + Index_file + "|wc -l").read())
    print("Total emails is " + str(total))
    X = [None] * total
    y = [None] * total
    X_result = [None] * 19178
    y_result = [None] * 19178
    # tmp_X = [None] * total
    # y = [None] * 19178
    XSpam = []
    XHam = []
    SignalSpam = 0
    SignalHam = 0
    counter = 0
    counter1 = 0
    max_email = 0
    with open(Index_file) as f:
        for line in f:
            if counter % 10000 == 0:
                print("split_data_set: Input Processing in Progress .......... ", str(counter))
            line = line.strip()
            label, key = line.split()
            fileName = key.split('/')[-1]
            newfileName = "trec07p/catemail_out/" + fileName
            with open(newfileName, 'r') as outputF:
                # tmp_X[counter] = outputF.read()
                X[counter] = outputF.read()
                if ((len(X[counter].split()) <= 100) and (len(X[counter].split()) >= 10)):
                    X_result[counter1] = X[counter]
                    if label.lower() == 'spam':
                        y_result[counter1] = 1
                        SignalSpam = 1
                    elif label.lower() == 'ham':
                        y_result[counter1] = 0
                        SignalHam = 1
                    # if y[counter] is None:
                        # print(X[counter])
                    counter1 += 1
                else:
                    max_email += 1
                counter +=1

    print("split_data_set: the number of emails are deleted : " + str(max_email))
    print("split_data_set: the number of valid emails: " + str(counter1))

    # Tham added
    # fileName = key.split('/')[-1]
    # newfileName = "trec07p/catemail_out_tham/" + fileName
    # X[counter1] = load(newfileName)
    # counter1 = counter1 + 1


    # X = X[:19178]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
    # return X_train, X_test,y_train, y_test
    return X_result, y_result

#---------------------------------------------------------------------
# calling split_data_set function to split the dataset into train_data and test_data
# result
#   X_train
#   X_test
#   y_train
#   y_test
#---------------------------------------------------------------------
# X_train, X_test, y_train, y_test = split_data_set(LABELS_FILE_OUTPUT)

X_result, y_result  = split_data_set(LABELS_FILE_OUTPUT)

X_train, X_test, y_train, y_test = train_test_split(X_result, y_result, test_size=0.30, random_state=2)

#---------------------------------------------------------------------
# create the document-term-matrix
# result
#   X_train_vector
#   X_test_vector
#---------------------------------------------------------------------
# vectorizerSpam = CountVectorizer()
# vectorizerHam = CountVectorizer()

# Trainvectorizer = CountVectorizer(decode_error='replace', stop_words='english', min_df=0.001)
Trainvectorizer = CountVectorizer()

X_train_vector = Trainvectorizer.fit_transform(X_train)

print("X_train_vector length = {}".format(X_train_vector.shape))
print('X_train_vector[1:11]: ')
print(X_train_vector[1:11])

TrainVocabulary = Trainvectorizer.get_feature_names()

print("TrainVocabulary length = {}".format(len(TrainVocabulary)))

res = dict(zip(Trainvectorizer.get_feature_names(), mutual_info_classif(X_train_vector, y_train, discrete_features=True)))

print('res: ', str(len(res)))
# print(res)

a = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
# print('a: ')
# print(a)
print('Total number of features in a: ', str(len(a)))

# sys.exit()



# import json
#
# with open("email_total_vocab_infor_gain.json", "w") as write_file:
#     json.dump(a, write_file)


# file_name_1 = "email_total_vocab_infor_gain.txt"
# print("....................................", file=open(file_name_1, "a"))
# print(a, file=open(file_name_1, "a"))
# print("....................................", file=open(file_name_1, "a"))

print('Total number of features: ', str(len(a)))
# print(len(a)) #number of features

# sys.exit()
# Choose vector length
# vec = 2000
listMIPub = list(a.keys())[:vec]
first_layer_parameter = vec

def reform_data_set(Index_file):
    total = int(os.popen("cat " + Index_file + "|wc -l").read())
    print("total is " + str(total))
    X = [None] * (total)
    y = [None] * (total)
    XSpam = []
    XHam = []
    SignalSpam = 0
    SignalHam = 0
    counter = 0
    max_email = 0
    count_spam =0
    count_ham = 0
    with open(Index_file) as f:
        for line in f:
            if counter % 10000 == 0:
                print("reform_data_set: Input Processing in Progress .......... ", str(counter))
            line = line.strip()
            label, key = line.split()
            if label.lower() == 'spam':
                y[counter] = 1
                SignalSpam = 1
                count_spam +=1
            elif label.lower() == 'ham':
                y[counter] = 0
                SignalHam = 1
                count_ham +=1

            fileName = key.split('/')[-1]
            newfileName = "trec07p/catemail_out/" + fileName
            X[counter] = load(newfileName)
            counter = counter + 1
    print("no of spam = ", str(count_spam))
    print('no of ham =', str(count_ham))
    # print("Reform dataset: the number of mails are deleted : " + str(max_email))
    print("Reform dataset: the number of total email: " + str(counter))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2) #,  (to reproduce result)
    return X_train, X_test,y_train, y_test, X


def load(path):
    email_text = ""
    with open(path, 'r') as test_file:
        email_text = test_file.read()

    # Tokenize the message
    tokens = nltk.word_tokenize(email_text)

    # Remove the tokens that are greater than 15 characters
    tokens = [x for x in tokens if ((x in listMIPub))]

    # Combine the tokens with space and build a single string again.
    return " ".join([i.strip() for i in tokens])


#--------------
# reform dataset
#---------------
X_train, X_test, y_train, y_test, X_reform = reform_data_set("trec07p/final_out/email_length.txt")

# sys.exit()

print("X_train[0]:")
print(X_train[0])
print("len(X_train[0]):")
print(len(X_train[0]))

print("X_test[0]:")
print(X_test[0])

print("len(X_test[0])")
print(len(X_test[0]))


#-----------------
vectorizer = CountVectorizer()

X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)
X_reform_vector = vectorizer.transform(X_reform)

print("after reform: X_train_vector: ")
print(X_train_vector.shape)
print(X_train_vector[1])





#
# print("The length of X_train_vector is: %s" % (str(X_train_vector.shape)), file=open(file_name, "a"))
# print("The length of X_test_vector is: %s" % (str(X_test_vector.shape)), file=open(file_name, "a"))
#

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

print(maxFrequence)
# round all feature values to integers and save time to a birary file .npy#
# X_train_vector = np.around((X_train_vector / maxFrequence ) * 255)
# X_test_vector = np.around((X_test_vector / maxFrequence ) * 255)
# X_reform_vector = np.around((X_reform_vector / maxFrequence ) * 255)
#
# print("after normalized: X_train_vector: ")
# print(X_train_vector.shape)
# print(X_train_vector[1])
#
# print("X_train_vector toarray: ")
# print(X_train_vector.shape)
# print(X_train_vector.toarray()[1].reshape(50,10))

# sys.exit()
# print(X_train_vector[0].shape)
# for i in X_train_vector.shape[0]:
#     print(X_train_vector[0][i])

# file1 = 'trec07-results/email_X_dataset_vector_length_{}_1'.format(vec)
# file2 = 'trec07_email_trained_model/email_X_test_vector_length_{0}_entire_{1}'.format(vec, d)
# file22 = 'trec07_email_trained_model/email_X_train_vector_length_{0}_entire_{1}'.format(vec, d)
# np.save(file1, X_test_vector.toarray()[1])
# np.save(file2, X_test_vector.toarray())
# np.save(file22, X_train_vector.toarray())
# np.save('trec07_email_trained_model/email_X_dataset_vector_length_20000_entire', X_reform_vector.toarray())

# np.save('trec07_email_trained_model/email_X_train_vector_length_2000_entire', X_train_vector.toarray())
# sys.exit()


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
encoded_Y = encode.transform(y_train) # one-hot encoding
dummy_y_train = np_utils.to_categorical(encoded_Y)
print('y_train[100]= ')
print(y_train[100])
print('dummy_y_train[100]= ')
print(dummy_y_train[100])

encode2 = LabelEncoder()
encode2.fit(y_test)
encoded_Y2 = encode2.transform(y_test)
dummy_y_test = np_utils.to_categorical(encoded_Y2)
print('dummy_y_test [1]=')
print(dummy_y_test[1])

file3 = 'trec07-results/email_dummy_y_test_X_{}_1'.format(vec)
# file4 = 'trec07_email_trained_model/email_dummy_y_test_X_2000_entire_%s' %d
# file3 = 'trec07_email_trained_model/email_dummy_y_train_X_2000_entire_%s' %d
np.save(file3, dummy_y_test[1])
# np.save(file4, dummy_y_test)
# np.save(file3, dummy_y_train)

# file4 = 'trec07_email_trained_model/email_y_test_X_{0}_entire_{1}'.format(vec, d)
# file3 = 'trec07_email_trained_model/email_y_train_X_{0}_entire_{1}'.format(vec, d)
# np.save(file3, dummy_y_test[1])
# np.save(file4, y_test)
# np.save(file3, y_train)



# sys.exit()

import time
import datetime

#---------------------------------------------------------------------
# create the ML model, linear -> X*X -> linear
# create factor to get the output
# result
#   model
#---------------------------------------------------------------------


print("The vector length = %s" % (str(vec)), file=open(file_name, "a"))

# model = keras.Sequential([
#     keras.layers.InputLayer(input_shape=(first_layer_parameter,)),
#     keras.layers.Dense(40),
#     keras.layers.Lambda(lambda x: x * x),
#     keras.layers.Dense(20),
#     keras.layers.Dense(2,activation='sigmoid')
# ])

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(first_layer_parameter,), name='input_layer'),
    keras.layers.Dense(40, name='hidden_1'),
    keras.layers.Lambda(lambda x: x * x, name='lambda_layer'),
    keras.layers.Dense(20, name='hidden_2'),
    keras.layers.Dense(10, name='hidden_3'),
    keras.layers.Dense(2,activation='softmax', name='output_layer') #keras.layers.Dense(2,activation='sigmoid', name='output_layer')
])


# model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
with open(file_name, 'a') as fi:
	print('Before_Quant_modelsummary:', file=fi)
	with redirect_stdout(fi):
		model.summary()
#print(model.summary())

# training phase #
# model.fit(X_train_vector.toarray(), dummy_y_train, epochs=20, verbose=1); # train the ML algorithm on training data: vector X and y
time1 = datetime.datetime.now()
history = model.fit(X_train_vector.toarray(), dummy_y_train, epochs=30, verbose=0); # train the ML algorithm on training data: vector X and y
time2 = datetime.datetime.now()
print("Training time = %s" % str(time2-time1), file=open(file_name, "a"))

# import pandas as pd
# history_frame = pd.DataFrame(history.history)
# print("....................................", file=open(file_name, "a"))
# for i in range(20):
#     print('loss in train : %f' % history_frame.loc[i, ['loss']], file=open(file_name, "a"))
#     print('accuracy in train : %f' % history_frame.loc[i, ['accuracy']], file=open(file_name, "a"))
# print("....................................", file=open(file_name, "a"))


#
# # # #######################
# print('----- after model.fit()')
layer_name = 'hidden_2'
intermediate_layer_model = keras.Model(inputs=model.input,
                                       outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model(X_reform_vector.toarray())

print('intermediate_output.shape= ')
print(intermediate_output.shape)

# # for i in range(intermediate_output.shape[0]):
# for i in range(intermediate_output.shape[0]):
#     # print(intermediate_output[i].numpy())
np.save('trec07_email_trained_model/hidden_2_output_entire_dataset', intermediate_output.numpy())
#
# ## X_train_set
intermediate_output_X_train = intermediate_layer_model(X_train_vector.toarray())
#
# # for i in range(intermediate_output.shape[0]):
# for i in range(intermediate_output_X_train.shape[1]):
#     # print(intermediate_output[i].numpy())
np.save('trec07_email_trained_model/hidden_2_output_X_train', intermediate_output_X_train.numpy())
#
# ## X_test_set
intermediate_output_X_test = intermediate_layer_model(X_test_vector.toarray())
#
# # for i in range(intermediate_output.shape[0]):
# for i in range(intermediate_output_X_test.shape[1]):
#     # print(intermediate_output[i].numpy())
np.save('trec07_email_trained_model/hidden_2_output_X_test', intermediate_output_X_test.numpy())
# # ##############3###########

#sys.exit()



# # prediction phase #
print("\nSTART EVALUATE THE MODEL ON TEST SET\n")
time1 = datetime.datetime.now()
loss, accuracy = model.evaluate(X_test_vector.toarray(), dummy_y_test, verbose=2); # model evaluation on test data
time2 = datetime.datetime.now()
print("Evaluate time = %s" % str(time2-time1), file=open(file_name, "a"))
print("\nFINISH EVALUATE THE MODEL ON TEST SET\n")
print('Loss before quant: %f' % loss, file=open(file_name, "a"))
print('Accuracy before quant: %f' % (accuracy*100), file=open(file_name, "a"))

modelPredictionTrain = model.predict(X_train_vector.toarray()); # prediction on train data
modelPredictionTest = model.predict(X_test_vector.toarray());  # prediction on test data


#--------------------------------------------------------------------
# convert the model weight to integer
# result
#   mail_model.tflite  -- file contains lite model
#   mail_model_quant.tflite -- file contains int model
#   mail_model_quant_io.tflite -- file contains only int model
#--------------------------------------------------------------------
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_models_dir = pathlib.Path(".")
# mail = tf.cast(X_train_vector.toarray(), tf.float32)
# mail_ds = tf.data.Dataset.from_tensor_slices((mail)).batch(1)
# def representative_data_gen():
#   for input_value in mail_ds.take(33125):
#     yield [input_value]
#
# converter.representative_dataset = representative_data_gen
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
#
# tflite_model_quant = converter.convert()
# trained_model_name = "trec07-results/email_model_quant_io_{0}_40_20_10_2_.tflite".format(vec)
# tflite_model_quant_file = tflite_models_dir/trained_model_name
# tflite_model_quant_file.write_bytes(tflite_model_quant)

tflite_models_dir = pathlib.Path(".")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


# Save the unquantized/float model:
tflite_model_file = tflite_models_dir / "trec07_email_model_no_quant_io.tflite"
tflite_model_file.write_bytes(tflite_model)


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

# Save the quantized model:
tflite_model_quant_file = tflite_models_dir / "trec07_email_model_quant_io.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)


#---------------------------------------------------------------------------------
# test the new quant model
#---------------------------------------------------------------------------------
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter_quant.allocate_tensors()
# input_index_quant = interpreter_quant.get_input_details()[0]["index"]
# output_index_quant = interpreter_quant.get_output_details()[0]["index"]
# test_mail = np.expand_dims(X_test_vector.toarray()[1], axis=0).astype(np.float32) # no need
# interpreter_quant.set_tensor(input_index_quant, test_mail)
# interpreter_quant.invoke()
# predictions = interpreter_quant.get_tensor(output_index_quant)


#----------------------------------------------------------------------------------
# evaluate the models
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
time1 = datetime.datetime.now()
print("Accuracy after quant: %s" % (str(evaluate_model(interpreter_quant))), file=open(file_name, "a"))
time2 = datetime.datetime.now()
print("Evaluate time with quantized model = %s" % str(time2-time1), file=open(file_name, "a"))


interpreter = interpreter_quant
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
fi.close()

