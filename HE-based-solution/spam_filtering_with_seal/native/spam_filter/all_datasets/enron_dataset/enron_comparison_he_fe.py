"""
In this example Alice trains a spam classifier on some e-mails dataset she
owns. She wants to apply it to Bob's personal e-mails, without

1) asking Bob to send his e-mails anywhere
2) leaking information about the learned model or the dataset she has learned
from
3) letting Bob know which of his e-mails are spam or not.

Alice trains a spam classifier with logistic regression on some data she
possesses. After learning, she generates public/private key pair with a
Paillier schema. The model is encrypted with the public key. The public key and
the encrypted model are sent to Bob. Bob applies the encrypted model to his own
data, obtaining encrypted scores for each e-mail. Bob sends them to Alice.
Alice decrypts them with the private key to obtain the predictions spam vs. not
spam.

Example inspired by @iamtrask blog post:
https://iamtrask.github.io/2017/06/05/homomorphic-surveillance/

Dependencies: numpy, sklearn
"""

import time
import os.path
from zipfile import ZipFile
from urllib.request import urlopen
from contextlib import contextmanager

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import phe as paillier
import sys

np.random.seed(42)


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
from numpy import asarray
from numpy import savetxt
tf.compat.v1.enable_eager_execution()

# np.random.seed(42)
from datetime import date
from datetime import datetime
today = date.today()
d1 = today.strftime("%b-%d-%Y")
now = datetime.now()
d = now.strftime("%d-%m-%Y-%H:%M:%S")

vec = 5000
num_test = 1000
hidd_1 = 40
hidd_2 = 20
file_name = "enron_1-2_email_Prediction_IG_{0}_{1}_{2}_10_2_HE_more_emails.txt".format(vec, hidd_1, hidd_2)
print("++++++++++++++++++++++++++++++++++++++++++++++\n", file=open(file_name, "a"))
print("++++++++++++++++++++++++++++++++++++++++++++++\n", file=open(file_name, "a"))
print("vec = {}".format(vec), file=open(file_name, "a"))
LABELS_FILE_OUTPUT = 'index_enron_1_2.txt'


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
    #     total = int(os.popen("cat " + Index_file + "|wc -l").read())
    #     print("Total emails is " + str(total))
    X = [None] * 11029  # total emails
    y = [None] * 11029
    XSpam = []
    XHam = []
    SignalSpam = 0
    SignalHam = 0
    counter = 0
    counter1 = 0
    max_email = 0

    # ### run this for the first time only
    # file_name = "valid_email_from-10w_500w.txt"
    # count =0
    # with open(Index_file) as f:
    #     for line in f:
    #         line = line.strip()
    #         newfileName = "catemail_out/" + line
    #         with open(newfileName, 'r') as outputF:
    #             X[counter] = outputF.read()
    #             if ((len(X[counter].split()) <= 500) and (len(X[counter].split()) >= 10)):
    #                 print(line, file=open(file_name, "a"))
    #                 count +=1
    ####
    # count = 6748  # valid emails
    count = 10212
    X_result = [None] * count
    y_result = [None] * count

    with open(Index_file) as f:
        for line in f:
            if counter % 10000 == 0:
                print("split_data_set: Input Processing in Progress .......... ", str(counter))
            line = line.strip()
            label = line.split('.')[-2]
            #             fileName = key.split('/')[-1]
            newfileName = "catemail_out/" + line
            with open(newfileName, 'r') as outputF:
                X[counter] = outputF.read()
                if ((len(X[counter].split()) <= 500) and (len(X[counter].split()) >= 10)):
                    X_result[counter1] = X[counter]
                    if label.lower() == 'spam':
                        y_result[counter1] = 1
                        SignalSpam = 1
                    elif label.lower() == 'ham':
                        y_result[counter1] = 0
                        SignalHam = 1
                    counter1 += 1
                else:
                    max_email += 1
                counter += 1

    print("split_data_set: the number of emails are deleted : " + str(max_email))
    print("split_data_set: the number of valid emails: " + str(counter1))
    return X_result, y_result


# LABELS_FILE_OUTPUT = 'index_new'



def load(path):
    email_text = " "
    with open(path, 'r') as test_file:
        email_text = test_file.read()

    # Tokenize the message
    tokens = nltk.word_tokenize(email_text)


    tokens_new = [x for x in tokens if ((x in listMIPub))]

    return " ".join([i.strip() for i in tokens_new])

def reform_data_set(Index_file):
    X = [None] * 10212
    y = [None] * 10212
    XSpam = []
    XHam = []
    SignalSpam = 0
    SignalHam = 0
    counter = 0
    max_email = 0
    count_spam = 0
    count_ham = 0
    with open(Index_file) as f:
        for line in f:
            if counter % 10000 == 0:
                print("reform_data_set: Input Processing in Progress .......... ", str(counter))
            line = line.strip()
            label = line.split('.')[-2]
            if label.lower() == 'spam':
                y[counter] = 1
                SignalSpam = 1
                count_spam +=1
            elif label.lower() == 'ham':
                y[counter] = 0
                SignalHam = 1
                count_ham +=1

            fileName = line
            newfileName = "catemail_out/" + line
            X[counter] = load(newfileName)
            counter = counter + 1
    # print("Reform dataset: the number of mails are deleted : " + str(max_email))
    print("Reform dataset: the number of total email: " + str(counter))
    print("no of spam = ", str(count_spam))
    print('no of ham = ', str(count_ham))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)
    return X_train, X_test,y_train, y_test, X

X_result, y_result = split_data_set(LABELS_FILE_OUTPUT)
    # print(y_result[1:10])

X_train, X_test, y_train, y_test = train_test_split(X_result, y_result, test_size=0.30, random_state=2)
print('before reform: ')
print(X_train[1])

min_df = 1
print("mindf = {}".format(min_df), file=open(file_name, "a"))
Trainvectorizer = CountVectorizer(min_df=min_df)
X_train_vector = Trainvectorizer.fit_transform(X_train)
# print(X_train_vector[1])
# print(y_train[1:10])

TrainVocabulary = Trainvectorizer.get_feature_names()

res = dict(
    zip(Trainvectorizer.get_feature_names(), mutual_info_classif(X_train_vector, y_train, discrete_features=True)))
total_vocab_sort = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
print("total vocab: {}".format(len(total_vocab_sort)), file=open(file_name, "a"))

# import json

# with open("linspam_total_vocab_infor_gain.json", "w") as write_file:
#     json.dump(total_vocab_sort, write_file)

listMIPub = list(total_vocab_sort.keys())[:vec]

X_train, X_test, y_train_before, y_test_before, X_reform = reform_data_set("valid_email_from-10w_500w.txt")
print('after reform: ')
print(X_train[1])

vectorizer = CountVectorizer()
X_train_vector = vectorizer.fit_transform(X_train)
X_test_vector = vectorizer.transform(X_test)
# X_reform_vector = vectorizer.transform(X_reform)

# maxFrequenceT = np.amax(X_train_vector)
# maxFrequenceT1 = np.amax(X_test_vector)
# if (maxFrequenceT > maxFrequenceT1):
#     maxFrequence = maxFrequenceT
# else:
#     maxFrequence = maxFrequenceT1

# # print(maxFrequence)
# # round all feature values to integers
# X_train_vector = np.around((X_train_vector / maxFrequence) * 255)
# X_test_vector = np.around((X_test_vector / maxFrequence) * 255)
# # X_reform_vector = np.around((X_reform_vector / maxFrequence ) * 255)

print(X_test_vector[100])
print(X_train_vector[1000])

print("y_train len = ", len(y_train_before), file=open(file_name, "a"))
print("y_test len = ", len(y_test_before), file=open(file_name, "a"))



file2 = 'enron_X_test_vector_length_{0}_entire_{1}.csv'.format(vec, d)
file22 = 'enron_email_y_test_vector_length_{0}_entire_{1}.csv'.format(vec, d)
# file33 = 'ceas08_email_encoded_y_test_vector_length_{0}_entire_{1}.csv'.format(vec, d)

savetxt(file2, X_test_vector.toarray()[0:num_test],delimiter=',')


print('#======= with FE ============#', file=open(file_name, "a"))

first_layer_parameter = X_test_vector.shape[1]

#---------------------------------------------------------------------
# transfer the result to 2 type
# result
#   dummy_y_train
#   dummy_y_test
#   dummy_y_train_private
#   dummy_y_test_private
#---------------------------------------------------------------------
encode = LabelEncoder()
encode.fit(y_train_before)
encoded_Y = encode.transform(y_train_before)
dummy_y_train = np_utils.to_categorical(encoded_Y)
encode2 = LabelEncoder()
encode2.fit(y_test_before)
encoded_Y2 = encode2.transform(y_test_before)
dummy_y_test = np_utils.to_categorical(encoded_Y2)

savetxt(file22, y_test_before[0:num_test],delimiter=',')


# np.save('dummy_y_test_1', dummy_y_test[1])
# np.save('trec07_y_train_freq', y_train)
# np.save('trec07_y_test_freq', y_test)

#---------------------------------------------------------------------
# create the ML model, linear -> X*X -> linear
# create factor to get the output
# result
#   model
#---------------------------------------------------------------------
# model = keras.Sequential([
#     keras.layers.InputLayer(input_shape=(first_layer_parameter,)),
#     keras.layers.Dense(40),
#     keras.layers.Lambda(lambda x: x * x),
#     keras.layers.Dense(20)
# ])

# new model:

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(first_layer_parameter,), name='input_layer'),
    keras.layers.Dense(hidd_1, name='hidden_1'),
    keras.layers.Lambda(lambda x: x * x, name='lambda_layer'),
    keras.layers.Dense(hidd_2, name='hidden_2'),
    keras.layers.Dense(10, name='hidden_3'),
    keras.layers.Dense(2,activation='sigmoid', name='output_layer')
])


model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
print("----------------------------------------------------------------")
print("-----------------the Q task with 2 output------------------")
print(model.summary())
model.fit(X_train_vector.toarray(), dummy_y_train, epochs=30, verbose=0)
loss, accuracy = model.evaluate(X_test_vector.toarray(), dummy_y_test, verbose=0)
print('Accuracy before quant: %f' % (accuracy*100))
print('Accuracy before quant: %f' % (accuracy*100), file=open(file_name, "a"))
modelPredictionTrain = model.predict(X_train_vector.toarray())
modelPredictionTest = model.predict(X_test_vector.toarray())

print("model weights:")
for layer in model.layers:
    print(layer.get_weights())

# sys.exit()

#--------------------------------------------------------------------
# convert the model weight to integer
# result
#   mail_model.tflite  -- file contains lite model
#   mail_model_quant.tflite -- file contains int model
#   mail_model_quant_io.tflite -- file contains only int model
#--------------------------------------------------------------------
tflite_models_dir = pathlib.Path(".")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
# # Save the unquantized/float model:
# tflite_model_file = tflite_models_dir / "enron12_email_model_no_quant_io.tflite"
# tflite_model_file.write_bytes(tflite_model)


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
tflite_model_quant_file = tflite_models_dir/"enron_email_vector_length_{0}_model_quantized_io_{1}.tflite".format(vec,d)
tflite_model_quant_file.write_bytes(tflite_model_quant)


sys.exit()
## to float16
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
#
# tflite_models_dir = pathlib.Path(".")
# mail = tf.cast(X_train_vector.toarray(), tf.float32)
# mail_ds = tf.data.Dataset.from_tensor_slices((mail)).batch(1)
#
# def representative_dataset_gen():
#   for input_value in mail_ds.take(33125):
#     yield [input_value]
#
#
# converter.representative_dataset = representative_dataset_gen
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
# #                                        tf.lite.OpsSet.TFLITE_BUILTINS]
# converter.target_spec.supported_types = [tf.float16]
# tflite_model_quant = converter.convert()
# tflite_model_quant_file = tflite_models_dir/"enron12_email_model_quant_io_freq.tflite"
# tflite_model_quant_file.write_bytes(tflite_model_quant)


#---------------------------------------------------------------------------------
# test the new quant model
#---------------------------------------------------------------------------------
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter_quant.allocate_tensors()
# input_index_quant = interpreter_quant.get_input_details()[0]["index"]
# output_index_quant = interpreter_quant.get_output_details()[0]["index"]
# test_mail = np.expand_dims(X_test_vector.toarray()[1], axis=0).astype(np.float32)
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
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)
    prediction_digits = []
    for test_data in X_test_vector.toarray():
        test_data = np.expand_dims(test_data, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_data)
        interpreter.invoke()
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == y_test_before[index]:
            accurate_count += 1
    accuracy = accurate_count * 1.0 / len(prediction_digits)
    return accuracy * 100
X_test_vector.toarray()

print("Accuracy after quant: " + str(evaluate_model(interpreter_quant)))
print("Accuracy after quant: " + str(evaluate_model(interpreter_quant)), file=open(file_name, "a"))




########################################################


@contextmanager
def timer(no_runs, file_name):
    """Helper for measuring runtime"""

    time0 = time.perf_counter()
    yield
    print('[avg running time: %.4f s]' % ((time.perf_counter() - time0)/no_runs))
    print('[avg running time: %.4f s]' % ((time.perf_counter() - time0) / no_runs), file=open(file_name, "a"))

class Alice:
    """
    Trains a Logistic Regression model on plaintext data,
    encrypts the model for remote use,
    decrypts encrypted scores using the paillier private key.
    """

    def __init__(self):
        self.model = LogisticRegression()

    def generate_paillier_keypair(self, n_length):
        self.pubkey, self.privkey = \
            paillier.generate_paillier_keypair(n_length=n_length)

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def encrypt_weights(self):
        coef = self.model.coef_[0, :]
        print(coef)
        print(coef.shape)
        print(self.model.intercept_[0])
        encrypted_weights = [self.pubkey.encrypt(coef[i])
                             for i in range(coef.shape[0])]
        encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
        print("number of encrypted weights {}".format(coef.shape[0]))
        print(encrypted_weights)
        return encrypted_weights, encrypted_intercept

    def decrypt_scores(self, encrypted_scores):
        return [self.privkey.decrypt(s) for s in encrypted_scores]

    def decrypt_email(self, encrypted_email):
        return [self.privkey.decrypt(s) for s in encrypted_email]


class Bob:
    """
    Is given the encrypted model and the public key.

    Scores local plaintext data with the encrypted model, but cannot decrypt
    the scores without the private key held by Alice.
    """

    def __init__(self, pubkey):
        self.pubkey = pubkey

    def set_weights(self, weights, intercept):
        self.weights = weights
        self.intercept = intercept

    def encrypted_score(self, x):
        """Compute the score of `x` by multiplying with the encrypted model,
        which is a vector of `paillier.EncryptedNumber`"""
        score = self.intercept
        _, idx = x.nonzero()
        for i in idx:
            score += x[0, i] * self.weights[i]
        # print("encrypted score = {}".format(score[1:5]))
        return score

    def encrypted_evaluate(self, X):
        return [self.encrypted_score(X[i, :]) for i in range(X.shape[0])]


    def encrypt_email(self, email_vec):
        # coef = self.model.coef_[0, :]
        encrypted_email = [self.pubkey.encrypt(email_vec[:, i])
                             for i in range(email_vec.shape[1])]
        # encrypted_intercept = self.pubkey.encrypt(self.model.intercept_[0])
        print("number of elements in email vector {}".format(email_vec.shape[1]))
        return encrypted_email




if __name__ == '__main__':

    # tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    # sys.exit()

    X_result, y_result = split_data_set(LABELS_FILE_OUTPUT)
    # print(y_result[1:10])

    X_train, X_test, y_train, y_test = train_test_split(X_result, y_result, test_size=0.30, random_state=2)
    print('before reform: ')
    print(X_train[1])

    min_df = 1
    print("mindf = {}".format(min_df), file=open(file_name, "a"))
    Trainvectorizer = CountVectorizer(min_df=min_df)
    X_train_vector = Trainvectorizer.fit_transform(X_train)
    # print(X_train_vector[1])
    # print(y_train[1:10])

    TrainVocabulary = Trainvectorizer.get_feature_names()

    res = dict(
        zip(Trainvectorizer.get_feature_names(), mutual_info_classif(X_train_vector, y_train, discrete_features=True)))
    total_vocab_sort = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    print("total vocab: {}".format(len(total_vocab_sort)), file=open(file_name, "a"))

    # import json

    # with open("linspam_total_vocab_infor_gain.json", "w") as write_file:
    #     json.dump(total_vocab_sort, write_file)

    listMIPub = list(total_vocab_sort.keys())[:vec]

    X_train, X_test, y_train_before, y_test_before, X_reform = reform_data_set("valid_email_from-10w_500w.txt")
    print('after reform: ')
    print(X_train[1])

    vectorizer = CountVectorizer()
    X_train_vector = vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)
    # X_reform_vector = vectorizer.transform(X_reform)

    maxFrequenceT = np.amax(X_train_vector)
    maxFrequenceT1 = np.amax(X_test_vector)
    if (maxFrequenceT > maxFrequenceT1):
        maxFrequence = maxFrequenceT
    else:
        maxFrequence = maxFrequenceT1

    # print(maxFrequence)
    # round all feature values to integers
    X_train_vector = np.around((X_train_vector / maxFrequence) * 255)
    X_test_vector = np.around((X_test_vector / maxFrequence) * 255)
    # X_reform_vector = np.around((X_reform_vector / maxFrequence ) * 255)

    print(X_test_vector[100])
    print(X_train_vector[1000])

    print("y_train len = ", len(y_train_before), file=open(file_name, "a"))
    print("y_test len = ", len(y_test_before), file=open(file_name, "a"))


    X = X_train_vector
    y = [None] * len(y_train_before)

    for i in range(len(y_train_before)):
        if y_train_before[i] == 0:
            y[i] = -1
        else:
            y[i] = y_train_before[i]

    X_test = X_test_vector
    # y_test = y_test
    y_test = [None] * len(y_test_before)

    for i in range(len(y_test_before)):
        if y_test_before[i] == 0:
            y_test[i] = -1
        else:
            y_test[i] = y_test_before[i]

    print("Training set: {}".format(X.shape))
    print("Test set: {}".format(X_test.shape))
    no_runs = X_test.shape[0]
    print("X_train[100]:")
    print(X[100])
    print("y_train[100]:")
    print(y[100])
    print("X_test[1]:")
    print(X_test[1])
    print("y_test[1]:")
    print(y_test[1])

    # print('#======= with HE paillier ============#')
    # print('#======= with HE paillier ============#', file=open(file_name, "a"))
    #
    print("Alice: Generating paillier keypair", file=open(file_name, "a"))
    alice = Alice()
    # NOTE: using smaller keys sizes wouldn't be cryptographically safe
    alice.generate_paillier_keypair(n_length=1024)

    # test_email_vec = X_test[1, :]
    # print("test_email_vec shape: ")
    # print(test_email_vec.shape[1])
    # # print(test_email_vec)
    # # print(test_email_vec.toarray())




    # sys.exit()

    #
    # print("Alice: Learning spam classifier", file=open(file_name, "a"))
    # with timer(1, file_name) as t:
    #     alice.fit(X, y)
    #
    # # print("Classify with model in the clear -- "
    # #       "what Alice would get having Bob's data locally", file=open(file_name, "a"))
    # # with timer(no_runs, file_name) as t:
    # #     error = np.mean(alice.predict(X_test) != y_test)
    # # print("label from plaintext = {}".format(alice.predict(X_test)))
    # # print("Prediction on plaintext Error {:.3f}".format(error))
    # # print("test accuracy on plaintext data = {}".format(100 * (1 - error)), file=open(file_name, "a"))
    #
    # print("Alice: Encrypting classifier", file=open(file_name, "a"))
    # with timer(1, file_name) as t:
    #     encrypted_weights, encrypted_intercept = alice.encrypt_weights()
    #
    # # bob = Bob(alice.pubkey)
    # # print("Sender: encrypting the vector of the email:")
    # # print("Sender: encrypting the vector of the email: ", file=open(file_name, "a"))
    # # encrypted_email = bob.encrypt_email(test_email_vec.toarray().astype("float32"))
    #
    # for i in range(5):
    #     print("Alice: Encrypting classifier", file=open(file_name, "a"))
    #     with timer(1, file_name) as t:
    #         encrypted_weights, encrypted_intercept = alice.encrypt_weights()
    #     print("Bob: decrypt the encrypted email vector: ")
    #     print("Bob: decrypt the encrypted email vector: ", file=open(file_name, "a"))
    #     with timer(1, file_name) as t:
    #         alice.decrypt_email(encrypted_weights)
    #
    # # print(alice.decrypt_email(encrypted_weights))
    #
    # sys.exit()

    # print("Bob: Scoring with encrypted classifier",file=open(file_name, "a"))
    # bob = Bob(alice.pubkey)
    # bob.set_weights(encrypted_weights, encrypted_intercept)
    # with timer(no_runs, file_name) as t:
    #     encrypted_scores = bob.encrypted_evaluate(X_test)
    #
    # print("Alice: Decrypting Bob's scores", file=open(file_name, "a"))
    # with timer(no_runs, file_name) as t:
    #     scores = alice.decrypt_scores(encrypted_scores)
    # error = np.mean(np.sign(scores) != y_test)
    # # print("decrypted label = {}".format(scores))
    # print("Prediction on encrypted data Error {:.3f} -- this is not known to Alice, who does not possess "
    #       "the ground truth labels".format(error))
    # print("test accuracy on encrypted data = {}".format(100*(1-error)), file=open(file_name, "a"))
    #
    # sys.exit()


