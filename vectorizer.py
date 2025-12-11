from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

# Get the current directory of the script to avoid path errors
cur_dir = os.path.dirname(__file__)

# Load the stop words (now located in the 'data' folder)
# We use os.path.join for cross-platform compatibility (Windows/Mac/Linux)
stop = pickle.load(open(os.path.join(cur_dir, 'data', 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    """
    Cleans and tokenizes the input text.
    Removes HTML tags and emoticons, keeping only words.
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
           + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Initialize the HashingVectorizer
# This is used to convert text into a vector of numbers for the model
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)