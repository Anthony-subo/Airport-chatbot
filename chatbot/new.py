import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
load_dotenv()  # This will load variables from a .env file if it exists


# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer and load intents
lemmatizer = WordNetLemmatizer()
intents = json.loads(open(r'C:\python project\chatbot\intents.json').read())

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Process each intent in the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Add the pattern and associated tag to documents
        documents.append((word_list, intent['tag']))
        # Add the tag to classes if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Sort classes
classes = sorted(set(classes))

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create the training set
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)
    \


    

# Shuffle the training data and convert to numpy arrays
random.shuffle(training)
training = np.array(training)

# Split the training data into inputs (X) and outputs (y)
train_x = training[:, :len(words)]
train_y = training[:, len(words):]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')
print('Done')
# set OPENAI_API_KEY=your-openai-api-key-here

# verify echo $Env:OPENAI_API_KEY chatbot\Scripts\activate  echo %OPENAI_API_KEY%   python app.py
