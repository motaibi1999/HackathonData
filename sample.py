import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def compute_sentiment_vader(sent):
    sent_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sent_obj.polarity_scores(sent)
    if sentiment_dict['compound'] >= 0.05 :
        return("Positive")
 
    elif sentiment_dict['compound'] <= - 0.05 :
        return("Negative")
 
    else :
        return("Neutral")


if __name__ == '__main__':
    df = pd.read_csv("data1.csv")
    df['sentiment'] = df['review'].apply(lambda x: compute_sentiment_vader(str(x)))
    print(df.head())
    df.to_csv('file1.csv')
    print(compute_sentiment_vader("sample sentence"))
    

