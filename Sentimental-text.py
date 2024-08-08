from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment > 0.6:
        return "Very positive"
    elif sentiment > 0.2:
        return "Slightly positive"
    elif sentiment == 0:
        return "Neutral"
    elif sentiment > -0.55:
        return "Slightly negative"
    else:
        return "Very negative"

# Take input from the user
text = input("Enter a sentence for sentiment analysis: ")

# Analyze the sentiment and print the result
result = analyze_sentiment(text)
print(result)
