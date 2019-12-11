
""" 
coding: utf-8

Text-Speech Generation from Image Using Neural Networks

"""

import csv
import pickle
import operator
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from numpy.random import seed
from string import punctuation

from keras.layers import Input
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.text import Tokenizer
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Dense, Concatenate, Dropout, Embedding, add
from keras.layers.recurrent import LSTM, GRU, SimpleRNN

"""
Loading the text data
"""
textfile_train = open("./Flickr8k.token.txt", "r")
doc = textfile_train.read()

pd.set_option('display.max_colwidth', -1)
df = pd.read_csv('Flickr8k.token.txt', sep="\t", names=['Image_Name','Image_Caption'],header=None)

new_df = df.copy()
new_df["Image_Name"]= new_df.iloc[:,0].str.slice(0,-2)


"""
Pre-processing all the text descriptions
"""
imageDescription = dict()

for rows in new_df.iterrows():
    if rows[1][0] not in imageDescription.keys():
        imageDescription[rows[1][0]] = [rows[1][1]]
    else:
        imageDescription[rows[1][0]].append(rows[1][1])

table = str.maketrans('', '', punctuation)
        
new_dict = {k: [sentence.lower() for sentence in v] for k, v in imageDescription.items()}
new_dict = {k: [sentence.translate(table) for sentence in new_dict[k]] for k in new_dict}
new_dict = {k:[' '.join([word for word in sentence.split(" ") if len(word)>1 and word.isalpha()]) for sentence in new_dict[k]]for k in new_dict}

vocabulary = set()
for key in new_dict.keys():
    [vocabulary.update(d.split()) for d in new_dict[key]]
print('Original Vocabulary Size: %d' % len(vocabulary))


"""
Removing all the words with frequence of appearance < 10
"""
captions=[]
word_count= dict()

for v in new_dict.values():
    for sentence in v:
        captions.append(sentence)

for sent in captions:
    word = sent.split(" ")
    for w in word: 
        if w in word_count:
            word_count[w] += 1
        else:
            word_count[w] = 1
            
vocab = [w for w in word_count if word_count[w] >= 10]

"""
Plotting the word counts
"""
sorted_d = pd.DataFrame(sorted(word_count.items(), key=operator.itemgetter(1),reverse=True))
sorted_d.columns = ["Word","Word_Count"]
s = sorted_d.head(50)
s.plot(x="Word",y="Word_Count",kind='barh',legend=None).invert_yaxis()
plt.xlabel("Words")
plt.ylabel("Word Counts")
plt.yticks(rotation=0)

sns.set(style="whitegrid")
plt.figure(figsize=(20,6))
sns.barplot(x=s["Word"],y=s["Word_Count"])
plt.xlabel("Words")
plt.ylabel("Word Counts")
plt.xticks(rotation='vertical')


"""
Building dictionary wordtoindex and indextoword which consists of
unique words and the corresponding index
"""
wordtoindex = {} #returns index of the word
indextoword = {} #returns word given the index

index = 1
for word in vocab:
    wordtoindex[word] = index
    indextoword[index] = word
    index+=1
    

"""
Seperating the date into train, validation and test sets
"""
pd.set_option('display.max_colwidth', -1)
train_df = pd.read_csv('Flickr_8k.trainImages.txt', names=['Image_Name'],header=None)
test_df = pd.read_csv('Flickr_8k.testImages.txt', names=['Image_Name'],header=None)
dev_df = pd.read_csv('Flickr_8k.devImages.txt', names=['Image_Name'],header=None)
train_dict = dict()

for i in range(len(train_df)):
    if train_df.iloc[i][0] in new_dict.keys():
        caption_list = new_dict.get(train_df.iloc[i][0])
        for sentence in caption_list:
            s ='startseq ' + (sentence) + ' endseq'
            if train_df.iloc[i][0] not in train_dict.keys():
                train_dict[train_df.iloc[i][0]] = [s]
            else:
                train_dict[train_df.iloc[i][0]].append(s)


"""
Adding a 'startseq' prefix and an 'endseq' postfix to each description
"""
dev_dict = dict()

for i in range(len(dev_df)):
    if dev_df.iloc[i][0] in new_dict.keys():
        caption_list = new_dict.get(dev_df.iloc[i][0])
        for sentence in caption_list:
            s ='startseq ' + (sentence) + ' endseq'
            if dev_df.iloc[i][0] not in dev_dict.keys():
                dev_dict[dev_df.iloc[i][0]] = [s]
            else:
                dev_dict[dev_df.iloc[i][0]].append(s)


"""
Load image features from their respective pickle files
"""
encoded_train = pickle.load(open("encoded_train_images.pkl","rb"))
encoded_test = pickle.load(open("encoded_test_images.pkl","rb"))
encoded_validation = pickle.load(open("encoded_dev_images.pkl","rb"))


"""
Finding the max length of a caption in the train set, this is used
to limit the description while generating it.
"""
captionsList =[]

for l in train_dict.values():
    for i in l:
        captionsList.append(i)


maxLendesc = max(len(c.split(" ")) for c in captionsList)
print("Max Description length : ",maxLendesc)


"""
Creating incremental sequences and target for each caption 
"""
tokenizer = Tokenizer()
t = tokenizer.fit_on_texts(captionsList)
vocab_size = len(tokenizer.word_index)+1

def createSequence(tokenizer, maxLendesc,train_dict, images, vocab_size):
    X1, X2, y = list(), list(), list()
    for image, captions in train_dict.items():
        for line in captions:
            seq = tokenizer.texts_to_sequences([line])[0]
            for i in range(1,len(seq)):
                input_seq, output_seq = seq[:i], seq[i]
                input_seq = pad_sequences([input_seq], maxlen=maxLendesc)[0] #dtype='int32', padding='post', truncating='post', value=0.0)[0]
                output_seq = to_categorical([output_seq], num_classes=vocab_size)[0]#, dtype='int32')[0]
                X1.append(images[image])
                X2.append(input_seq)
                y.append(output_seq)
    return np.array(X1), np.array(X2), np.array(y)
    
X1_train, X2_train, y_train = createSequence(tokenizer, maxLendesc,train_dict, encoded_train, vocab_size)
X1_test, X2_test, y_test = createSequence(tokenizer, maxLendesc,dev_dict, encoded_validation, vocab_size)


"""
Varying model architecture by using different models like SimpleRNN, GRU, LSTM
"""
vocab_size = len(tokenizer.word_index)+1 

def model_RNN_FFNN(maxLendesc,vocab_size):
    # Image feature extraction
    in1 = Input(shape=(2048,))
    featureExtracted1 = Dropout(0.5)(in1)
    featureExtracted2 = Dense(256, activation='relu')(featureExtracted1)
    
    # Captions
    in2 = Input(shape=(maxLendesc,))
    sentenceExtraction1 = Embedding(vocab_size, 256, mask_zero=True)(in2) # input_dim = vocab_size, output_dim = 256
    sentenceExtraction2 = Dropout(0.5)(sentenceExtraction1)
    sentenceExtraction3 = SimpleRNN(256)(sentenceExtraction2)
    
    # Input to the feed forward NN
    ff1 = add([featureExtracted2,sentenceExtraction3])
    ff2 = Dense(256, activation="relu")(ff1)
    ff_out = Dense(vocab_size, activation="softmax")(ff2)
    
    model = Model(inputs=[in1,in2], outputs=ff_out) # All the inputs required to compute the output
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) #Configures the model for training
    
    print(model.summary())
    return model

def model_GRU_FFNN(maxLendesc,vocab_size):
    # Image feature extraction
    in1 = Input(shape=(2048,))
    featureExtracted1 = Dropout(0.5)(in1)
    featureExtracted2 = Dense(256, activation='relu')(featureExtracted1)
    
    # Captions
    in2 = Input(shape=(maxLendesc,))
    sentenceExtraction1 = Embedding(vocab_size, 256, mask_zero=True)(in2) # input_dim = vocab_size, output_dim = 256
    sentenceExtraction2 = Dropout(0.5)(sentenceExtraction1)
    sentenceExtraction3 = GRU(256)(sentenceExtraction2)
    
    # Input to the feed forward NN
    ff1 = add([featureExtracted2,sentenceExtraction3])
    ff2 = Dense(256, activation="relu")(ff1)
    ff_out = Dense(vocab_size, activation="softmax")(ff2)
    
    model = Model(inputs=[in1,in2], outputs=ff_out) # All the inputs required to compute the output
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) #Configures the model for training
    
    print(model.summary())
    return model

def model_LSTM_FFNN(maxLendesc,vocab_size):
    # Image feature extraction
    in1 = Input(shape=(2048,))
    featureExtracted1 = Dropout(0.5)(in1)
    featureExtracted2 = Dense(256, activation='relu')(featureExtracted1)

    # Captions
    in2 = Input(shape=(maxLendesc,))
    sentenceExtraction1 = Embedding(vocab_size, 256, mask_zero=True)(in2) # input_dim = vocab_size, output_dim = 256
    sentenceExtraction2 = Dropout(0.5)(sentenceExtraction1)
    sentenceExtraction3 = LSTM(256)(sentenceExtraction2)

    # Input to the feed forward NN
    ff1 = add([featureExtracted2,sentenceExtraction3])
    ff2 = Dense(256, activation="relu")(ff1)
    ff_out = Dense(vocab_size, activation="softmax")(ff2)

    model = Model(inputs=[in1,in2], outputs=ff_out) # All the inputs required to compute the output
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) #Configures the model for training

    print(model.summary())
    return model
    
    
model = model_LSTM_FFNN(maxLendesc,vocab_size)  
model.fit([X1_train, X2_train], y_train, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1_test, X2_test], y_test))

# Saving the model to a file
file = "./model.sav"
pickle.dump(model, open(file, 'wb'))

"""
Predicting the captions for each test image
"""
def get_word(prediction, tokenizer):
    word = tokenizer.index_word[prediction]
    return word

def get_description(model, photo_feature, tokenizer, max_word_len):
    desc =  'startseq'
    for i in range(max_word_len):
        word_vector = tokenizer.texts_to_sequences([desc])[0]
        word_vector = pad_sequences([word_vector], maxlen=max_word_len)#, dtype='int32', padding='post', truncating='post', value=0.0)#[0]
        prediction = model.predict([[photo_feature], word_vector], verbose=0)
        prediction = np.argmax(prediction)
        word = get_word(prediction, tokenizer)
        desc += " "+str(word)
        if word == 'endseq':
            break
        i += 1
    return desc

#filename = './model.sav'
#model = pickle.load(open(filename, 'rb'))

encoded_test = pickle.load(open("encoded_test_images.pkl","rb"))
index = 0
for image in encoded_test.keys():
    print("Image: "+str(image))
    print("Actual Prediction:")
    print(test_dict[image])
    print("Our Prediction")
    print(get_description(model, encoded_test[image], tokenizer, maxLendesc))


"""
Calculating the BLEU metric for each model
"""

def evaluate_model(model, descriptions, photos, tokenizer, max_len):
    actual, predicted = list(), list()
    for key, desc_list in descriptions.items():
        predict = get_description(model, photos[key], tokenizer, max_len)
        refs = [d.split() for d in desc_list]
        actual.append(refs)
        predicted.append(predict.split())
        
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
    
evaluate_model(model, test_dict, encoded_test, tokenizer, maxLendesc)
