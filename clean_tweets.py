import pandas as pd
import numpy as np

# print("Training Set:" % tweets.columns, tweets.shape, len(tweets))
# print(tweets)

# excitement_data = pd.DataFrame(excitement_tweets, columns=['clean_text'])  # 'id',
# happy_data = pd.DataFrame(happy_tweets, columns=['clean_text'])
# excitement_data = pd.DataFrame(excitement_tweets, columns=['clean_text'])
# excitement_data = pd.DataFrame(excitement_tweets, columns=['clean_text'])
# excitement_data = pd.DataFrame(excitement_tweets, columns=['clean_text'])
# excitement_data = pd.DataFrame(excitement_tweets, columns=['clean_text'])
# print(tweets_data)

# test = pd.read_csv('test_tweets.csv')
# print("Test Set:"% test.columns, test.shape, len(test))
#
import sys
import os
import nltk

# nltk.download('punkt')
import csv
import datetime
from bs4 import BeautifulSoup
import re
import itertools
import emoji


#
#
# #####################################################################################
# #
# # DATA CLEANING
# #
# #####################################################################################

def tweet_cleaning(tweet):
    # Escaping HTML characters
    # tweet = BeautifulSoup(tweet).get_text()

    # Special case not handled previously.

    # tweet = tweet.replace('\x92', "'")

    # Removal of hastags/account
    # tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", tweet).split())

    # Removal of address
    # tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())

    # Removal of Punctuation
    # tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", tweet).split())

    # tweet = re.sub('@[A-Za-z0–9]+', '', tweet)  # Removing @mentions
    # tweet = re.sub('#', '', tweet)  # Removing '#' hash tag
    # tweet = re.sub('RT[\s]+', '', tweet)  # Removing RT(retweets)
    # tweet = re.sub('https?:\/\/\S+', '', tweet)  # Removing hyperlink

    # Lower case
    # tweet = tweet.lower()

    # CONTRACTIONS source: https://en.wikipedia.org/wiki/Contraction_%28grammar%29
    # CONTRACTIONS = load_dict_contractions()
    # tweet = tweet.replace("’", "'")
    # words = tweet.split()
    # reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    # tweet = " ".join(reformed)
    # tweet = tweet.replace("’", "'")
    # words = tweet.split()
    # tweet = " ".join(words)

    # Standardizing words
    # tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))

    # Deal with emoticons source: https://en.wikipedia.org/wiki/List_of_emoticons
    # SMILEY = load_dict_smileys()
    # words = tweet.split()
    # reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    # tweet = " ".join(reformed)
    # words = tweet.split()
    # tweet = " ".join(words)

    # Deal with emojis
    # tweet = emoji.demojize(tweet)

    # tweet = tweet.replace(":", " ")
    # tweet = ' '.join(tweet.split())

    return tweet


# excitement
e_data = pd.read_csv('data/collect_excitement_tweets.csv')
e_tweets = pd.DataFrame(e_data, columns=['clean_text'])
e_clean_tweets = pd.DataFrame()
e_clean_tweets['clean_tweets'] = np.array([tweet_cleaning(tweet) for tweet in e_tweets['clean_text']])
e_clean_tweets['label'] = 'excitement'
e_clean_tweets.to_csv('emotions/excitement_tweets.csv', index=False)

# happy
h_data = pd.read_csv('data/collect_happy_tweets.csv')
h_tweets = pd.DataFrame(h_data, columns=['clean_text'])
h_clean_tweets = pd.DataFrame()
h_clean_tweets['clean_tweets'] = np.array([tweet_cleaning(tweet) for tweet in h_tweets['clean_text']])
h_clean_tweets['label'] = 'happy'
h_clean_tweets.to_csv('emotions/happy_tweets.csv', index=False)

# pleasant
p_data = pd.read_csv('data/collect_pleasant_tweets.csv')
p_tweets = pd.DataFrame(p_data, columns=['clean_text'])
p_clean_tweets = pd.DataFrame()
p_clean_tweets['clean_tweets'] = np.array([tweet_cleaning(tweet) for tweet in p_tweets['clean_text']])
p_clean_tweets['label'] = 'pleasant'
p_clean_tweets.to_csv('emotions/pleasant_tweets.csv', index=False)

# surprise
s_data = pd.read_csv('data/collect_surprise_tweets.csv')
s_tweets = pd.DataFrame(s_data, columns=['clean_text'])
s_clean_tweets = pd.DataFrame()
s_clean_tweets['clean_tweets'] = np.array([tweet_cleaning(tweet) for tweet in s_tweets['clean_text']])
s_clean_tweets['label'] = 'surprise'
s_clean_tweets.to_csv('emotions/surprise_tweets.csv', index=False)

# fear
f_data = pd.read_csv('data/collect_fear_tweets.csv')
f_tweets = pd.DataFrame(f_data, columns=['clean_text'])
f_clean_tweets = pd.DataFrame()
f_clean_tweets['clean_tweets'] = np.array([tweet_cleaning(tweet) for tweet in f_tweets['clean_text']])
f_clean_tweets['label'] = 'fear'
f_clean_tweets.to_csv('emotions/fear_tweets.csv', index=False)

# angry
a_data = pd.read_csv('data/collect_angry_tweets.csv')
a_tweets = pd.DataFrame(a_data, columns=['clean_text'])
a_clean_tweets = pd.DataFrame()
a_clean_tweets['clean_tweets'] = np.array([tweet_cleaning(tweet) for tweet in a_tweets['clean_text']])
a_clean_tweets['label'] = 'angry'
a_clean_tweets.to_csv('emotions/angry_tweets.csv', index=False)

# crowdsourcing input data
# cs_input = pd.concat([e_clean_tweets.head(20), h_clean_tweets.head(20)], axis=0, ignore_index=True)
# print(e_clean_tweets.head(20))
# print(cs_input)
cs_input = e_clean_tweets.head(20).append(h_clean_tweets.head(20)).append(p_clean_tweets.head(20)).append(
    s_clean_tweets.head(20)).append(f_clean_tweets.head(20)).append(a_clean_tweets.head(20), ignore_index=True)
print(cs_input)
cs_input.to_csv('cs_input.csv', index=False)


# print(clean_tweets)

# clean_tweets = clean_tweets.drop_duplicates('clean_text')


#
#
# #####################################################################################
# #
# # DATA PROCESSING
# #
# #####################################################################################
#
# def transform_instance(row):
#     cur_row = []
#     # Prefix the index-ed label with __label__
#     label = "__label__" + row[4]
#     cur_row.append(label)
#     cur_row.extend(nltk.word_tokenize(tweet_cleaning(row[2].lower())))
#     return cur_row


#
#
# def preprocess(input_file, output_file, keep=1):
#     i = 0
#     with open(output_file, 'w') as csvoutfile:
#         csv_writer = csv.writer(csvoutfile, delimiter=' ', lineterminator='\n')
#         with open(input_file, 'r', newline='', encoding='latin1') as csvinfile:  # ,encoding='latin1'
#             csv_reader = csv.reader(csvinfile, delimiter=',', quotechar='"')
#             for row in csv_reader:
#                 if row[4] != "MIXED" and row[4].upper() in ['POSITIVE', 'NEGATIVE', 'NEUTRAL'] and row[2] != '':
#                     row_output = transform_instance(row)
#                     csv_writer.writerow(row_output)
#                     # print(row_output)
#                 i = i + 1
#                 if i % 10000 == 0:
#                     print(i)
#
#
# # Preparing the training dataset
# preprocess('betsentiment-EN-tweets-sentiment-teams.csv', 'tweets.train')
#
# # Preparing the validation dataset
# preprocess('betsentiment-EN-tweets-sentiment-players.csv', 'tweets.validation')
#
#
# #####################################################################################
# #
# # UPSAMPLING
# #
# #####################################################################################
#
# def upsampling(input_file, output_file, ratio_upsampling=1):
#     # Create a file with equal number of tweets for each label
#     #    input_file: path to file
#     #    output_file: path to the output file
#     #    ratio_upsampling: ratio of each minority classes vs majority one. 1 mean there will be as much of each class than there is for the majority class
#
#     i = 0
#     counts = {}
#     dict_data_by_label = {}
#
#     # GET LABEL LIST AND GET DATA PER LABEL
#     with open(input_file, 'r', newline='') as csvinfile:
#         csv_reader = csv.reader(csvinfile, delimiter=',', quotechar='"')
#         for row in csv_reader:
#             counts[row[0].split()[0]] = counts.get(row[0].split()[0], 0) + 1
#             if not row[0].split()[0] in dict_data_by_label:
#                 dict_data_by_label[row[0].split()[0]] = [row[0]]
#             else:
#                 dict_data_by_label[row[0].split()[0]].append(row[0])
#             i = i + 1
#             if i % 10000 == 0:
#                 print("read" + str(i))
#
#     # FIND MAJORITY CLASS
#     majority_class = ""
#     count_majority_class = 0
#     for item in dict_data_by_label:
#         if len(dict_data_by_label[item]) > count_majority_class:
#             majority_class = item
#             count_majority_class = len(dict_data_by_label[item])
#
#             # UPSAMPLE MINORITY CLASS
#     data_upsampled = []
#     for item in dict_data_by_label:
#         data_upsampled.extend(dict_data_by_label[item])
#         if item != majority_class:
#             items_added = 0
#             items_to_add = count_majority_class - len(dict_data_by_label[item])
#             while items_added < items_to_add:
#                 data_upsampled.extend(
#                     dict_data_by_label[item][:max(0, min(items_to_add - items_added, len(dict_data_by_label[item])))])
#                 items_added = items_added + max(0, min(items_to_add - items_added, len(dict_data_by_label[item])))
#
#     # WRITE ALL
#     i = 0
#
#     with open(output_file, 'w') as txtoutfile:
#         for row in data_upsampled:
#             txtoutfile.write(row + '\n')
#             i = i + 1
#             if i % 10000 == 0:
#                 print("writer" + str(i))
#
#
# upsampling('tweets.train', 'uptweets.train')
# # No need to upsample for the validation set. As it does not matter what validation set contains.
#
#
# #####################################################################################
# #
# # TRAINING
# #
# #####################################################################################
#
# # Full path to training data.
# training_data_path = 'uptweets.train'
# validation_data_path = 'tweets.validation'
# model_path = ''
# model_name = "model-en"
#
#
# def train():
#     print('Training start')
#     try:
#         hyper_params = {"lr": 0.01,
#                         "epoch": 20,
#                         "wordNgrams": 2,
#                         "dim": 20}
#
#         print(str(datetime.datetime.now()) + ' START=>' + str(hyper_params))
#
#         # Train the model.
#         model = fastText.train_supervised(input=training_data_path, **hyper_params)
#         print("Model trained with the hyperparameter \n {}".format(hyper_params))
#
#         # CHECK PERFORMANCE
#         print(str(datetime.datetime.now()) + 'Training complete.' + str(hyper_params))
#
#         model_acc_training_set = model.test(training_data_path)
#         model_acc_validation_set = model.test(validation_data_path)
#
#         # DISPLAY ACCURACY OF TRAINED MODEL
#         text_line = str(hyper_params) + ",accuracy:" + str(model_acc_training_set[1]) + ", validation:" + str(
#             model_acc_validation_set[1]) + '\n'
#         print(text_line)
#
#         # quantize a model to reduce the memory usage
#         model.quantize(input=training_data_path, qnorm=True, retrain=True, cutoff=100000)
#
#         print("Model is quantized!!")
#         model.save_model(os.path.join(model_path, model_name + ".ftz"))
#
#         ##########################################################################
#         #
#         #  TESTING PART
#         #
#         ##########################################################################
#         model.predict(['why not'], k=3)
#         model.predict(['this player is so bad'], k=1)
#
#     except Exception as e:
#         print('Exception during training: ' + str(e))
#
#
# # Train your model.
# train()

def read_tweets(csvfile):
    tweets = pd.read_csv(csvfile)
    tweets_data = pd.DataFrame(tweets, columns=['clean_text'])
    clean_tweets = pd.DataFrame()
    clean_tweets['clean_tweets'] = np.array([tweet_cleaning(tweet) for tweet in tweets_data['clean_text']])
    clean_tweets['label'] = 'excitement'
    # print(clean_tweets)

    return clean_tweets

#
# excitement_tweets = pd.read_csv('data/collect_excitement_tweets.csv')
# happy_tweets = pd.read_csv('data/collect_happy_tweets.csv')
# pleasant_tweets = pd.read_csv('data/collect_pleasant_tweets.csv')
# surprise_tweets = pd.read_csv('data/collect_surprise_tweets.csv')
# fear_tweets = pd.read_csv('data/collect_fear_tweets.csv')
# angry_tweets = pd.read_csv('data/collect_angry_tweets.csv')
#
# read_tweets(excitement_tweets).to_csv('emotions/excitement_tweets.csv')
# # read_tweets(happy_tweets).to_csv('emotions/happy_tweets.csv')
# read_tweets(pleasant_tweets).to_csv('emotions/pleasant_tweets.csv')
# read_tweets(surprise_tweets).to_csv('emotions/surprise_tweets.csv')
# read_tweets(fear_tweets).to_csv('emotions/fear_tweets.csv')
# read_tweets(angry_tweets).to_csv('emotions/angry_tweets.csv')
