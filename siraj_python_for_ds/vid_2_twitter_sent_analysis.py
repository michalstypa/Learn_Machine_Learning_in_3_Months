import tweepy
import numpy as np
from textblob import TextBlob

consumer_key = 'zHwpAwATx5huctEaO2omWCVQi'
consumer_secret = 'rcH22IYVTc7lztE4jNzDqJriF4M1UlknqDQdisyygjqw2jyrXg'

access_token = '631255004-X2RTbQqjhl4Loy4KXGKyJrMr3WEK9DVkdFhkE4gw'
access_token_secret = 'z189clA1s0G9rgEwsjkbePlwQhZfXeJ9vqojQbpacGngR'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

public_tweets = api.search('Trump', count=100)

tweets = []

for tweet in public_tweets:
  tweets.append(TextBlob(tweet.text))

polarity = [t.sentiment.polarity for t in tweets]
subjectivity = [t.sentiment.subjectivity for t in tweets]

print(np.average(subjectivity))
print(np.average(subjectivity))