import os
from pathlib import Path
import pickle

import pandas as pd
from deep_translator import GoogleTranslator
from flask import Flask, jsonify, request
from flask_basicauth import BasicAuth
from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
from textblob import TextBlob

columns = ['tamanho', 'ano', 'garagem']

# df = pd.read_csv(os.path.join(path, 'casas.csv'))
# df = df[columns]

# x = df.drop('preco', axis=1)
# y = df['preco']

# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.3, random_state=42)
# model = LinearRegression()
# model.fit(x_train, y_train)

# pickle.dump(model, open('model.sav', 'wb'))

model = pickle.load(open('../../model.sav', 'rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)


@app.route('/')
def home():
    return 'My first API.'


@app.route('/feeling/<phrase>')
@basic_auth.required
def feeling(phrase):
    treated_phrase = str(phrase).strip()
    to_en = GoogleTranslator(
        source='auto', target='en').translate(treated_phrase)
    tb = TextBlob(to_en)
    polarity = tb.sentiment.polarity
    return f'phrase: {to_en}. Polarity: {polarity}'


@app.route('/quotation/', methods=['POST'])
@basic_auth.required
def quotation():
    data = request.get_json()
    data_input = [data[col] for col in columns]
    price = model.predict([data_input])
    return jsonify(preco=price[0])


app.run(debug=True, host='0.0.0.0')
