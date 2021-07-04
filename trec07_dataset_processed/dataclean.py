import string
import email
import nltk
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import html2text
import warnings


DATA_DIR = 'trec07p/data/'
LABELS_FILE = 'trec07p/full/index'
TEST_FILE = 'trec07p/data/inmail.28'

spam = 0
ham = 0
nltk.download('stopwords')
nltk.download('punkt')

with open(LABELS_FILE) as fp:
    Lines = fp.readlines()
    for line in Lines:
        if "spam" in line:
            spam += 1
        else:
            ham += 1

print("Number of spam: ", str(spam), "-",str(round(spam*100/(spam+ham),2)),"%")
print("Number of ham: ", str(ham), "-",str(round(ham*100/(spam+ham),2)),"%")

#%matplotlib inline

label=["Spam", "Ham"]
values=[spam, ham]
index = np.arange(len(label))
#plt.bar(label,values)
#plt.show()

punctuations = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()


# Combine the different parts of the email into a flat list of strings
def flatten_to_string(parts):
    ret = []
    if type(parts) == str:
        ret.append(parts)
    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)
    elif parts.get_content_type == 'text/plain':
        ret += parts.get_payload()
    return ret


# Extract subject and body text from a single email file
def extract_email_text(path):
    # Load a single email from an input file
    with open(path, errors='ignore') as f:
        msg = email.message_from_file(f)
    if not msg:
        return ""

    # Read the email subject
    subject = msg['Subject']
    if not subject:
        subject = ""

    # Read the email body
    body = ' '.join(m for m in flatten_to_string(msg.get_payload())
                    if type(m) == str)
    if not body:
        body = ""

    h = html2text.HTML2Text()
    h.ignore_links = True
    NoHtmlbody = h.handle(body)
    

    return subject + ' ' + NoHtmlbody


# Process a single email file into stemmed tokens
def load(path, TEST_MODE):
    email_text = ""

    if TEST_MODE == 0:
        email_text = extract_email_text(path)
    elif TEST_MODE == 1:
        with open(path, 'r') as test_file:
            email_text = test_file.read()

    if not email_text:
        with open(path,'r') as f:
            email_text = f.read()
        #return []
        

    # Remove non-alphabet characters
    email_text = re.sub('[^0-9a-zA-Z]+', ' ', email_text)
    # Normalize white spaces
    email_text = " ".join(email_text.split())

    # Change to lower case
    email_text = email_text.lower()

    # Tokenize the message
    tokens = nltk.word_tokenize(email_text)

    # Remove punctuation from tokens
    tokens = [i.strip("".join(punctuations)) for i in tokens
              if i not in punctuations]

    # Remove the tokens that are greater than 15 characters
    tokens = [x for x in tokens if len(x) <= 15]

    # Remove the tokens that are less than 3 characters
    tokens = [x for x in tokens if len(x) >= 3]

    # Remove tokens that contain only numbers
    tokens = [x for x in tokens if not x.isnumeric()]

    # Stem the tokens
    tokens = [stemmer.stem(w) for w in tokens if w not in stopwords]

    # Combine the tokens with space and build a single string again.
    return " ".join([i.strip() for i in tokens])
    #return email_text

print("********************* Example Before pre-processing *********************")
with open(TEST_FILE) as fp:
    Lines = fp.read()
    print(Lines)


print("********************* Example email after pre-processing *********************")
print(load(TEST_FILE,0))



# We are disable some warnings coming in different python versions.
warnings.simplefilter(action='ignore', category=FutureWarning)

# Total number of data samples
TOTAL = spam + ham

# TRAINING_SET_RATIO constant defines the training and test set split ratio. We will be using 70% of the emails for
# training and 30% of the emails for testing.

TRAINING_SET_RATIO = 0.7

# Empty arrays for storing the data.
X = [None] * TOTAL
y = [None] * TOTAL

# Iterate through the label file
counter = 0
with open(LABELS_FILE) as f:
    for line in f:
        if counter % 10000 == 0:
            print("Input Processing in Progress .......... ", str(counter), " out of ", str(TOTAL))
        line = line.strip()
        label, key = line.split()

        if label.lower() == 'ham':
            y[counter] = 1
        elif label.lower() == 'spam':
            y[counter] = 0

        fileName = key.split('/')[-1]
        newfileName = "trec07p/catemail_out_tham/" + fileName
        X[counter] = load(key, 0)
        with open(newfileName, 'w') as text_file:
            text_file.write(X[counter])
        counter = counter + 1






print("")
print("Verifying the dataset sizes")
print("---------------------------")
print("No. of samples in X", str(len(X)))
print("No. of labels in y", str(len(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2)

print("")
print("Verifying training and test set sizes")
print("-------------------------------------")
print("Training set size", str(len(X_train)))
print("Test set size", str(len(X_test)))


