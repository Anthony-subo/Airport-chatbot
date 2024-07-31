from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from keras.models import load_model
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import random
import openai

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sk-proj-LJCtqPoZIhgZme6US6FaT3BlbkFJ2oJ9Y0p028gXn8W5NrNU'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False, unique=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

# Initialize database
with app.app_context():
    db.create_all()

# Load trained model and other necessary files
model = load_model('chatbot_model.h5')
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# OpenAI API key
openai.api_key = 'sk-proj-pN6mtRCQanGFrwLQ3x2IT3BlbkFJXZbkFu7S93MSkZketGGE'  # Replace with your OpenAI API key

# Function to process user input and get response
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(ints, intents_json, language='english'):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            if language == 'kiswahili' and 'kiswahili_response' in i:
                result = random.choice(i['kiswahili_response'])
            else:
                result = random.choice(i['responses'])
            break
    return result

def get_chatgpt_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = response.choices[0].text.strip()
    return message

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Login failed. Check your username and password.', 'danger')
    return render_template('Login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(fullname=fullname, email=email, username=username, password=hashed_password)
        db.session.add(new_user)
        try:
            db.session.commit()
            flash('Signup successful! Please log in.', 'success')
            return redirect(url_for('home'))
        except:
            db.session.rollback()
            flash('Username or email already exists. Please choose another.', 'danger')
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.form['user_input']
    language = request.form['language'] if 'language' in request.form else 'english'  # Default to English if language not specified
    ints = predict_class(user_message)
    if not ints or float(ints[0]['probability']) < 0.5:
        res = get_chatgpt_response(user_message)
    else:
        res = get_response(ints, intents, language=language)
    return jsonify({'response': res})

if __name__ == '__main__':
    app.run(debug=True)