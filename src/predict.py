import joblib
import sys

def predict_sentiment(text):
    try:
        model = joblib.load('models/sentiment_model.pkl')
        prediction = model.predict([text])[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        return sentiment
    except FileNotFoundError:
        return "Model not found. Run train.py first."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        result = predict_sentiment(input_text)
        print(f"Text: {input_text}")
        print(f"Sentiment: {result}")
