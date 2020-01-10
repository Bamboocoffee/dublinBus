from django.shortcuts import render

# Create your views here.

import twitter_credentials
from tweepy import OAuthHandler
from tweepy import API
import json

class TweetFinder():

    """
    A class that opens a connection to the Twitter API
    Authentical credentials referenced from twitter_credentials.py
    
    Method get_tweets() collects recent (non-RT) tweets from
    Dublin Bus News. Returns a JSON, representing multi-dimensional
    dict, each higher order key a tweet, lower order keys tweet attributes.
    """

    def __init__(self, consumer_key, consumer_secret,
            access_token, access_token_secret):

        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret

        self.auth = OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)

    def get_tweets(self):

        """
        A method to fetch recent tweets from Dublin Bus News Twitter
        Stream.

        Returns JSON object with fields:
            [Recent, non-retweet, tweets with attributes] -- [
                time of tweet
                link to user to profile photo
                username
                user screen name
                user url
                tweet full text
                ]
        """

        api = API(self.auth)

        tweet_response = api.user_timeline(id='@dublinbusnews',
        tweet_mode="extended")

        tweet_list = {}
        i = 0

        for tweet in tweet_response:
            if tweet.full_text[0] != '@':
                i += 1
                time = tweet.created_at
                time = time.strftime("%Y-%m-%d %H:%M:%S")
                picture_link = tweet.user.profile_image_url
                username = tweet.user.name
                display_name = '@' + tweet.user.screen_name
                text = tweet.full_text
                user_url = tweet.user.url
                tweet_list[i] = {
                    'time':time,
                    'picture_link': picture_link,
                    'username': username,
                    'display_name': display_name,
                    'text': text,
                    'user_url':user_url
                }

        json_response = json.dumps(tweet_list)

        return json_response


def main():

    twitter_finder = TweetFinder(
        twitter_credentials.CONSUMER_KEY,
        twitter_credentials.CONSUMER_SECRET,
        twitter_credentials.ACCESS_TOKEN,\
        twitter_credentials.ACCESS_TOKEN_SECRET)

    return twitter_finder.get_tweets()


if __name__ == "__main__":

    main()
   
