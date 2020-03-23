import csv
import os
import pandas as pd
import tweepy
import re
import string
from textblob import TextBlob
import preprocessor as p
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import twitter_credentials

# Twitter credentials for the app

# pass twitter credentials to tweepy

auth = tweepy.OAuthHandler(twitter_credentials.CONSUMER_KEY, twitter_credentials.CONSUMER_SECRET)
auth.set_access_token(twitter_credentials.ACCESS_TOKEN, twitter_credentials.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

# file location changed to "data/" for clearer path
collect_keywords_tweets = "data/collect_keywords_tweets.csv"

collect_emotion_tweets = "data/collect_emotions_tweets.csv"

collect_excitement_tweets = "data/collect_excitement_tweets.csv"
collect_happy_tweets = "data/collect_happy_tweets.csv"
collect_pleasant_tweets = "data/collect_pleasant_tweets.csv"
collect_surprise_tweets = "data/collect_surprise_tweets.csv"
collect_fear_tweets = "data/collect_fear_tweets.csv"
collect_angry_tweets = "data/collect_angry_tweets.csv"

# columns of the csv file
COLS = ['id', 'created_at', 'source', 'original_text', 'clean_text', 'sentiment', 'polarity', 'subjectivity', 'lang',
        'favorite_count', 'retweet_count', 'original_author', 'possibly_sensitive', 'hashtags',
        'user_mentions', 'place', 'place_coord_boundaries']

# set two date variables for date range
# start_date = '2018-10-01'
# end_date = '2018-10-31'

# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])

# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])

# Emoji patterns
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

# combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)


# mrhod clean_tweets()
def clean_tweets(tweet):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)

    # after tweepy preprocessing the colon left remain after removing mentions
    # or RT sign in the beginning of the tweet
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)

    # remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)

    # filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []

    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)
    # print(word_tokens)
    # print(filtered_sentence)


# method write_tweets()
def write_tweets(keyword, file):
    # If the file exists, then read the existing data from the CSV file.
    if os.path.exists(file):
        df = pd.read_csv(file, header=0)
    else:
        df = pd.DataFrame(columns=COLS)
    # page attribute in tweepy.cursor and iteration
    for page in tweepy.Cursor(api.search, q=keyword,
                              count=200, include_rts=False).pages(10):
        for status in page:
            new_entry = []
            status = status._json

            # check whether the tweet is in english or skip to the next tweet
            if status['lang'] != 'en':
                continue

            # when run the code, below code replaces the retweet amount and
            # no of favorires that are changed since last download.
            if status['created_at'] in df['created_at'].values:
                i = df.loc[df['created_at'] == status['created_at']].index[0]
                if status['favorite_count'] != df.at[i, 'favorite_count'] or \
                        status['retweet_count'] != df.at[i, 'retweet_count']:
                    df.at[i, 'favorite_count'] = status['favorite_count']
                    df.at[i, 'retweet_count'] = status['retweet_count']
                continue

            # tweepy preprocessing called for basic preprocessing
            clean_text = p.clean(status['text'])

            # call clean_tweet method for extra preprocessing
            filtered_tweet = clean_tweets(clean_text)

            # pass textBlob method for sentiment calculations
            blob = TextBlob(filtered_tweet)
            Sentiment = blob.sentiment

            # seperate polarity and subjectivity in to two variables
            polarity = Sentiment.polarity
            subjectivity = Sentiment.subjectivity

            # new entry append
            new_entry += [status['id'], status['created_at'],
                          status['source'], status['text'], filtered_tweet, Sentiment, polarity, subjectivity,
                          status['lang'],
                          status['favorite_count'], status['retweet_count']]

            # to append original author of the tweet
            new_entry.append(status['user']['screen_name'])

            try:
                is_sensitive = status['possibly_sensitive']
            except KeyError:
                is_sensitive = None
            new_entry.append(is_sensitive)

            # hashtagas and mentiones are saved using comma separted
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in status['entities']['hashtags']])
            new_entry.append(hashtags)
            mentions = ", ".join([mention['screen_name'] for mention in status['entities']['user_mentions']])
            new_entry.append(mentions)

            # get location of the tweet if possible
            try:
                location = status['user']['location']
            except TypeError:
                location = ''
            new_entry.append(location)

            try:
                coordinates = [coord for loc in status['place']['bounding_box']['coordinates'] for coord in loc]
            except TypeError:
                coordinates = None
            new_entry.append(coordinates)

            single_tweet_df = pd.DataFrame([new_entry], columns=COLS)
            df = df.append(single_tweet_df, ignore_index=True)
            csvFile = open(file, 'a', encoding='utf-8')
    df.to_csv(csvFile, mode='a', columns=COLS, index=False, encoding="utf-8")
    # df.to_csv(csvFile, mode='a', columns=['original text'], index=False, encoding="utf-8")


# declare keywords as a query for three categories
telemedicine_keywords = '#telemedicine OR #telehealth OR #digitalhealth OR #ehealth OR #digitalpatient OR #digitaltransformation'
Epilepsy_keywords = '#Epilepsy OR #epilepsyawareness OR #epilepsyaction OR #epilepsyalerts OR #epilepsybed OR #epilepsycongres OR #epilepsysurgery OR #epilepsysurgery OR #Epilepsytreatment OR #seizures OR #seizurefree'
HeartDisease_keywords = '#HeartDisease OR #stroke OR #Stroking OR #strokepatient OR #StrokeSurvivor OR #hearthealth OR #Stroke OR #HeartFailure'

query_keywords = 'world OR work OR education OR disease OR COVID-19 OR coronavirus OR BillGates'
query_hash_tags = '#excited OR #excitement OR #exciting OR #happy OR #joy OR #love OR #pleasant OR #nice OR #cheerful ' \
                  'OR #surprise OR #sad OR #frustrated OR #fear OR #scared OR #afraid OR #disgusted OR #depressed OR ' \
                  '#angry OR #mad OR #annoyed '

excitement_hash_tags = '#excited OR #excitement OR #exciting'
happy_hash_tags = '#happy OR #joy OR #love '
pleasant_hash_tags = '#pleasant OR #nice OR #cheerful'
surprise_hash_tags = '#surprise OR #sad OR #frustrated'
fear_hash_tags = '#fear OR #scared OR #afraid OR #disgusted OR #depressed'
angry_hash_tags = '#angry OR #mad OR #annoyed'

# call main method passing keywords and file path

# write_tweets(query_keywords, collect_keywords_tweets)

# write_tweets(query_hash_tags, collect_emotion_tweets)
write_tweets(excitement_hash_tags, collect_excitement_tweets)
write_tweets(happy_hash_tags, collect_happy_tweets)
write_tweets(pleasant_hash_tags, collect_pleasant_tweets)
write_tweets(surprise_hash_tags, collect_surprise_tweets)
write_tweets(fear_hash_tags, collect_fear_tweets)
write_tweets(angry_hash_tags, collect_angry_tweets)

# with open('data/collect_keywords_tweets.csv', 'rb') as csvfile:
#     reader = csv.reader(csvfile)
#     reader = pd.np.array(reader)
#     tweets_text = pd.read_csv(csvfile)
# print(tweets_text)
