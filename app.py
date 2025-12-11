from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# Import our custom vectorizer from vectorizer.py
from vectorizer import vect

app = Flask(__name__)

######## PREPARING THE CLASSIFIER ########
cur_dir = os.path.dirname(__file__)
# Loading the pre-trained SGDClassifier
clf = pickle.load(open(os.path.join(cur_dir, 'data', 'pkl_objects', 'classifier.pkl'), 'rb'))
db = os.path.join(cur_dir, 'data', 'reviews.sqlite')

def classify(document):
    """
    Predicts the sentiment of a given text (document).
    Returns: label ('positive' or 'negative') and the probability score.
    """
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    """
    Updates the model with new user feedback (Online Learning).
    """
    X = vect.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    """
    Saves the user review into the SQLite database for future reference.
    """
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review, sentiment, date) VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

######## FLASK CONFIGURATION ########

class ReviewForm(Form):
    moviereview = TextAreaField('', [validators.DataRequired(), validators.Length(min=15)])

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        y, proba = classify(review)
        return render_template('results.html',
                               content=review,
                               prediction=y,
                               probability=round(proba*100, 2))
    return render_template('reviewform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    # Convert label to integer for the model (1 = Positive, 0 = Negative)
    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    
    # If the user says the prediction was wrong, flip the label
    if feedback == 'Incorrect':
        y = int(not(y))
    
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)