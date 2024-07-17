#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter

# Download NLTK resources (you may need to run this once)
nltk.download('vader_lexicon')

# Sample social media data (replace with your actual data collection method)
social_media_data = [
    "I love the new iPhone, it's amazing!",
    "This product is terrible, would not recommend.",
    "Just bought a new laptop and I'm very happy with it.",
    "The customer service was rude and unhelpful.",
    "Excited about the new movie release!",
    "Can't believe how bad the service is here.",
]

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Analyze sentiment of each text and collect scores
sentiment_scores = []
for text in social_media_data:
    scores = sid.polarity_scores(text)
    sentiment_scores.append(scores)

# Calculate overall sentiment
compound_scores = [score['compound'] for score in sentiment_scores]
overall_sentiment = sum(compound_scores) / len(compound_scores)

# Visualize sentiment distribution
positive_count = sum(1 for score in compound_scores if score > 0)
neutral_count = sum(1 for score in compound_scores if score == 0)
negative_count = sum(1 for score in compound_scores if score < 0)

labels = ['Positive', 'Neutral', 'Negative']
sizes = [positive_count, neutral_count, negative_count]
colors = ['#77DD77', '#FFD700', '#FF6961']

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Sentiment Distribution')
plt.axis('equal')
plt.show()

# Visualize sentiment over time (in this case, just an example)
dates = ['2024-06-01', '2024-06-02', '2024-06-03', '2024-06-04', '2024-06-05']
sentiment_values = [0.2, 0.3, -0.1, 0.5, -0.2]  # Replace with actual sentiment values

plt.figure(figsize=(10, 5))
plt.plot(dates, sentiment_values, marker='o', linestyle='-', color='b')
plt.title('Sentiment Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




