# tweeter-sentiment--analysis
tweeter_sentiment_analysis_in nlp model

NOTE dataset is taken from kaggle
link- https://www.kaggle.com/datasets/kazanova/sentiment140

Twitter Sentiment Analysis with Logistic Regression Overview
This project performs binary sentiment analysis on Twitter data using Logistic Regression. 
The model classifies tweets as either positive or negative based on their textual content.

🗃 Dataset
Source: tweeter.csv
Size: 1.6 million tweets
Classes:
0 → Negative sentiment
4 (converted to 1) → Positive sentiment

Requirements
libraries:-
import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
