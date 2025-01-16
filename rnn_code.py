import nltk
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pdfplumber  # Import the pdfplumber library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle


def calculate_perplexity(model, xs, ys):
    """
    Calculates the perplexity of a language model on a given dataset.

    Args:
      model: The trained language model.
      xs: The input sequences (predictor variables).
      ys: The true labels (one-hot encoded).

    Returns:
      The perplexity score.
    """
    total_log_prob = 0
    num_predictions = 0

    for i in range(len(xs)):
        input_seq = xs[i:i+1]  # Get a single input sequence
        target_seq = ys[i]  # Get the corresponding target sequence

        # Pad the input sequence if necessary
        padded_input_seq = pad_sequences(input_seq, maxlen=xs.shape[1], padding='pre') 

        predictions = model.predict(padded_input_seq)  # Get predictions
        
        # Calculate the cross-entropy loss for this sequence
        log_prob = -np.log(predictions[0][np.argmax(target_seq)])  
        
        total_log_prob += log_prob
        num_predictions += 1

    # Calculate perplexity
    perplexity = np.exp(total_log_prob / num_predictions)
    return perplexity

# Download 'punkt' for sentence tokenization
nltk.download('punkt')

# Function to clean the text (remove unwanted characters)
def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])
    return text

# PDF extraction logic using pdfplumber
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + " "  # Add a space after each page
        return full_text

# Load and preprocess the text data
pdf_path = "../data/the-secret-garden.pdf"  # Replace with the actual path to your PDF
text = extract_text_from_pdf(pdf_path)
text = clean_text(text)  # Clean the extracted text

# Sentence segmentation
sentences = nltk.sent_tokenize(text)

# Tokenization and Normalization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create input sequences (using 5-grams)
max_seq_length = 20  # Set a maximum sequence length
input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list) - max_seq_length + 1):
        n_gram_sequence = token_list[i:i+max_seq_length]
        input_sequences.append(n_gram_sequence)

# Pad sequences
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre'))

# ... (rest of your code) ...
# Create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

# One-hot encode the labels
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# Split the data
split_index_1 = int(len(xs) * 0.8)
split_index_2 = int(len(xs) * 0.9)
xs_train, xs_val, xs_test = xs[:split_index_1], xs[split_index_1:split_index_2], xs[split_index_2:]
ys_train, ys_val, ys_test = ys[:split_index_1], ys[split_index_1:split_index_2], ys[split_index_2:]

print("Data preprocessing completed.")



# Build the model
model = Sequential()
model.add(Embedding(total_words, 128, input_length=max_seq_length-1))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print("Model built successfully.")


# Hyperparameters are already defined in the model.compile() statement:
#   - Learning rate: 0.001 (default for Adam optimizer)
#   - Batch size: 64 (will be used in model.fit())
#   - Optimizer: Adam
#   - Epochs: Start with 10 (will be used in model.fit())

# Train the model
history = model.fit(xs_train, ys_train, epochs=10, batch_size=128,
                    validation_data=(xs_val, ys_val))

print("Model training completed.")



# Function to generate text with temperature
def generate_text(model, seed_text, next_words, temperature=1.0):
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


# Generate text with different temperatures
seed_text = "The secret garden"
next_words = 50

print("Generated text with temperature 0.7:")
print(generate_text(model, seed_text, next_words, temperature=0.7))

# Save the model
model.save("secret_garden_model.h5")  
print("Model saved to secret_garden_model.h5")