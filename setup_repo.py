import sqlite3
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# --- CONFIGURATION ---
dest_dir = os.path.join('data', 'pkl_objects')
db_path = os.path.join('data', 'reviews.sqlite')

def prepare_directories():
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")
    if not os.path.exists('data'):
        os.makedirs('data')

def create_stopwords():
    # A basic list of English stopwords to avoid NLTK dependency for this demo
    stop_words = {
        'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
        'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 
        'can', 'did', 'do', 'does', 'doing', 'don', 'down', 'during',
        'each', 'few', 'for', 'from', 'further',
        'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
        'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
        'just', 'me', 'more', 'most', 'my', 'myself',
        'no', 'nor', 'not', 'now',
        'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own',
        's', 'same', 'she', 'should', 'so', 'some', 'such',
        't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
        'this', 'those', 'through', 'to', 'too',
        'under', 'until', 'up',
        'very',
        'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
        'you', 'your', 'yours', 'yourself', 'yourselves'
    }
    
    stop_path = os.path.join(dest_dir, 'stopwords.pkl')
    pickle.dump(stop_words, open(stop_path, 'wb'), protocol=4)
    print(f"--> Stopwords saved to {stop_path}")
    return stop_words

def train_dummy_model(stop_words):
    # 1. Define tokenizer (Same logic as vectorizer.py)
    def tokenizer(text):
        return [w for w in text.split() if w not in stop_words]

    # 2. Create Vectorizer
    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2**21,
                             preprocessor=None,
                             tokenizer=tokenizer)

    # 3. Create Dummy Data (Just so the model exists and works)
    # In a real scenario, you would load the IMDb dataset here.
    X_train_text = [
        "I love this movie it is fantastic and great",
        "This film was terrible and boring",
        "Absolutely wonderful acting and plot",
        "I hated every minute of this bad movie",
        "Good movie",
        "Bad movie"
    ]
    y_train = np.array([1, 0, 1, 0, 1, 0]) # 1 = Positive, 0 = Negative

    X_train = vect.transform(X_train_text)

    # 4. Initialize and Train SGD Classifier
    clf = SGDClassifier(loss='log_loss', random_state=1)
    clf.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    # 5. Save the Classifier
    clf_path = os.path.join(dest_dir, 'classifier.pkl')
    pickle.dump(clf, open(clf_path, 'wb'), protocol=4)
    print(f"--> Dummy Model trained and saved to {clf_path}")

def create_database():
    if os.path.exists(db_path):
        os.remove(db_path) # Reset for clean install
    
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)')
    conn.commit()
    conn.close()
    print(f"--> Database initialized at {db_path}")

if __name__ == '__main__':
    print("Starting Project Setup...")
    prepare_directories()
    stops = create_stopwords()
    train_dummy_model(stops)
    create_database()
    print("\n✅ Setup Complete! You can now run 'python app.py'")