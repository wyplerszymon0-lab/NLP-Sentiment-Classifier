import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    data = {
        'text': [
            'I loved this movie, it was great!', 'Terrible film, a waste of time.',
            'Best acting I have ever seen.', 'Boring and predictable plot.',
            'Amazing visuals and sound.', 'I hated every minute of it.',
            'Pure masterpiece!', 'One of the worst movies this year.',
            'Highly recommended for everyone.', 'Not worth the ticket price.',
            'A truly delightful experience.', 'I fell asleep halfway through.',
            'The cinematography was breathtaking.', 'The script was poorly written.'
        ],
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    
    df = pd.DataFrame(data)
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['sentiment'], test_size=0.2, random_state=42
    )

    model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(classification_report(y_test, predictions))

    joblib.dump(model, 'models/sentiment_model.pkl')

if __name__ == "__main__":
    train_model()
