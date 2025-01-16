from flask import Flask, render_template, request  # Import request from Flask
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Download 'punkt' for sentence tokenization (if not already downloaded)
nltk.download('punkt')

# Load the saved model
model = load_model("secret_garden_model.h5")

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Get the maximum sequence length
max_seq_length = 20

# Function to generate text with temperature
def generate_text(model, tokenizer, seed_text, next_words, max_seq_length, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
        predicted_probs = model.predict(token_list)[0]  # Get predicted probabilities

        # Apply temperature
        predicted_probs = np.asarray(predicted_probs).astype('float64')
        predicted_probs = np.log(predicted_probs) / temperature
        exp_probs = np.exp(predicted_probs)
        predicted_probs = exp_probs / np.sum(exp_probs)

        predicted_id = np.argmax(np.random.multinomial(1, predicted_probs, 1))
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_id:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        seed_text = request.form["seed_text"]
        next_words = int(request.form["next_words"])
        temperature = float(request.form["temperature"])
        generated_text = generate_text(model, tokenizer, seed_text, next_words, max_seq_length, temperature)  # Pass all arguments
        return render_template("index.html", generated_text=generated_text)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False,port=5000)