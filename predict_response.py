# Text Data Preprocessing Lib
from data_preprocessing import get_stem_words
import tensorflow
import random
import numpy as np
import pickle
import json
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# words to be ignored/omitted while framing the dataset
ignore_words = ['?', '!', ',', '.', "'s", "'m"]


# Model Load Lib

# load the model
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl', 'rb'))
classes = pickle.load(open('./classes.pkl', 'rb'))


def preprocess_user_input(user_input):
    bag = [0] * len(words)

    # tokenize the user_input
    user_input_words = nltk.word_tokenize(user_input)

    # convert the user input into its root words: stemming
    user_input_words = [lemmatizer.lemmatize(
        word.lower()) for word in user_input_words]

    # Remove duplicacy and sort the user_input
    user_input_words = sorted(list(set(user_input_words)))

    # Input data encoding: Create BOW for user_input
    for w in user_input_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)


def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
    inp = inp.reshape(1, len(inp))

    prediction = model.predict(inp)
    predicted_class_label = np.argmax(prediction[0])

    return predicted_class_label


def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)
    predicted_class = classes[predicted_class_label]

    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            bot_response = random.choice(intent['responses'])
            return bot_response


print("Hi, I am Stella. How can I help you?")

while True:
    # take input from the user
    user_input = input('Type your message here: ')

    response = bot_response(user_input)
    print("Bot Response: ", response)
