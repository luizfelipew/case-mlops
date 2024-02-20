from flask import Flask
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv('casas.csv')
colunas = ['tamanho', 'preco']
df = df[colunas]
# Variavel explicativa -> x
X = df.drop('preco', axis=1)
# Variavel resposta -> y
y = df['preco']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

modelo = LinearRegression()
# ajuste do modelo apartir da minha base de treinamento
modelo.fit(X_train, y_train)


app = Flask(__name__)

@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt', to='en')
    polaridade = tb_en.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route('/cotacao/<int:tamanho>')
def cotacao(tamanho):
    preco = modelo.predict([[tamanho]])
    return str(preco) 

app.run(debug=True)