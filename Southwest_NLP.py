import re
import tweepy
import numpy as np
import pandas as pd
import json

import nltk
nltk.download('all')
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


#Variables that contain the credentials to access Twitter API
consumer_key    = ""
consumer_secret = ""
access_key      = ""
access_secret   = ""

# Setup access to API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

# create the API object
api = tweepy.API(auth, wait_on_rate_limit=True)

search_words = "@SouthwestAir"
search_query = search_words + "-filter:retweets AND -filter:replies"

limit = 20000000
tweets = tweepy.Cursor(api.search_tweets, q=search_query,lang="en", count = 200000, tweet_mode='extended').items(limit)

tweet_list_df = list()

for tweet in tweets:
    tweet_dict = {}
    tweet_dict['id'] = tweet.id
    tweet_dict['text'] = tweet.full_text
    tweet_dict['time'] = tweet.created_at
    tweet_dict['device'] = tweet.source
    tweet_dict['retweets'] = tweet.retweet_count
    tweet_dict['likes'] = tweet.favorite_count
    tweet_dict['location'] = tweet.user.location
    tweet_list_df.append(tweet_dict)
        

        
df_tweet = pd.DataFrame(tweet_list_df)

df_tweets = pd.read_csv('/Users/MY_USER_NAME/Desktop/SouthWest/Southwest_Twitterdata_20230111.csv', index_col=0)

# Data Labeling with LabelStudio
# Ticket Topics:
# Flight Changes & Cancellation
# Seats Help
# Baggage Help
# Rapid Rewards Help
# Refund Request & Status Help
# Covid19
# Others - help needed
# No response needed


# Load JSON data into a Python object
with open('/Users/MY_USER_NAME/Downloads/project-2-at-2023-01-16-06-16-abcfbb64.json', 'r') as f:
    data = json.load(f)

# create an empty list to store the data
text_result = []

# iterate through the data and append the text and result to the list
for item in data:
    text = item['data']['text']
    if 'annotations' in item and len(item['annotations'])>0:
        if 'result' in item['annotations'][0] and len(item['annotations'][0]['result'])>0:
            if 'value' in item['annotations'][0]['result'][0] and 'choices' in item['annotations'][0]['result'][0]['value'] and len(item['annotations'][0]['result'][0]['value']['choices'])>0:
                result = item['annotations'][0]['result'][0]['value']['choices'][0]
                text_result.append([text, result])

# create a dataframe from the list
df = pd.DataFrame(text_result, columns=['text', 'result'])




# create a list text
text = list(df['text'])

# preprocessing loop
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(text)):
    r = re.sub('[^a-zA-Z]', ' ', text[i])
    r = r.lower()
    r = r.split()
    r = [word for word in r if word not in stopwords.words('english')]
    r = [lemmatizer.lemmatize(word) for word in r]
    r = ' '.join(r)
    corpus.append(r)

#assign corpus to data['text']

df['text'] = corpus


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['result'], test_size=0.2)

# Extract features using TF-IDF
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Decision Tree Classifier
# Train a Decision Tree model
clf = DecisionTreeClassifier()
clf.fit(X_train_tfidf, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# Train a Random Forest model
clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(X_train_tfidf, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# Train a SVM model
clf = SVC(kernel='linear', C=1, gamma='scale')
clf.fit(X_train_tfidf, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Train a KNN classifier
clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
clf.fit(X_train_tfidf, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Train a Neural Network classifier
clf = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
clf.fit(X_train_tfidf, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)