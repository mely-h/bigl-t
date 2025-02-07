from transformers import pipeline

def analyze_sentiment(review):
    sentiment_analysis = pipeline("sentiment-analysis")
    max_length = 512
    tokens = review.split()

    sentiment = []
    current_chunk = []
    current_length = 0

    print(f"Traitement de l'avis : {review[:100]}...")

    for token in tokens:
        if current_length + len(token) + 1 > max_length:
            chunk = ' '.join(current_chunk)
            result = sentiment_analysis(chunk)
            sentiment.append(result[0]['label'])
            current_chunk = [token]
            current_length = len(token)
        else:
            current_chunk.append(token)
            current_length += len(token) + 1

    if current_chunk:
        chunk = ' '.join(current_chunk)
        result = sentiment_analysis(chunk)
        sentiment.append(result[0]['label'])

    if sentiment:
        final_sentiment = max(set(sentiment), key=sentiment.count)
        return final_sentiment
    
    return "UNKNOWN"
