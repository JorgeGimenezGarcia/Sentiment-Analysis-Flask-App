import pickle
import sqlite3
import numpy as np
import os
from vectorizer import vect  # Imports the HashingVectorizer logic

# Define logical paths for a robust execution
CUR_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(CUR_DIR, 'data', 'reviews.sqlite')
MODEL_PATH = os.path.join(CUR_DIR, 'data', 'pkl_objects', 'classifier.pkl')

def update_model(db_path, model_path, batch_size=10000):
    """
    Connects to the local SQLite database, fetches new verified reviews,
    and updates the SGDClassifier model (Online Learning).
    
    Args:
        db_path (str): Path to the SQLite database.
        model_path (str): Path to the pickled model file.
        batch_size (int): Number of rows to fetch at once.
    """
    print(f"--> Connecting to database at: {db_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load the existing model
    clf = pickle.load(open(model_path, 'rb'))
    print("--> Model loaded successfully. Starting update process...")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Select all data (in a real scenario, you might want a 'processed' flag column)
    c.execute('SELECT review, sentiment, date FROM review_db ORDER BY date DESC')
    
    results = c.fetchmany(batch_size)
    
    while results:
        data = np.array(results)
        X = data[:, 0]
        # Ensure sentiment is integer (0 or 1)
        y = data[:, 1].astype(int)
        
        classes = np.array([0, 1])
        
        # Transform text to vector
        X_train = vect.transform(X)
        
        # Update the model using partial_fit (Out-of-Core learning)
        clf.partial_fit(X_train, y, classes=classes)
        print(f"--> Processed batch of {len(X)} reviews.")
        
        results = c.fetchmany(batch_size)
    
    conn.close()
    
    # Save the updated model back to disk
    pickle.dump(clf, open(model_path, 'wb'), protocol=4)
    print("--> Model updated and saved successfully.")

if __name__ == '__main__':
    # This block allows the script to be run directly from the terminal
    if os.path.exists(DB_PATH) and os.path.exists(MODEL_PATH):
        update_model(DB_PATH, MODEL_PATH)
    else:
        print("CRITICAL ERROR: Database or Model file missing.")
        print(f"Check paths:\nDB: {DB_PATH}\nModel: {MODEL_PATH}")