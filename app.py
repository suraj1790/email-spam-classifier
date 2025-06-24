from flask import Flask,render_template,request
import pickle
import nltk
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords
nltk.download("stopwords")

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))



app = Flask(__name__)


@app.route("/",methods=["GET","POST"])
def home():
    return render_template("home.html")


def transform_text(text):
    # convert onto lowercase
    text = text.lower()

    # word tokinization
    text = nltk.word_tokenize(text)

    # remove penchuation and stopwords
    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    text = y.copy()
    y.clear()

    for word in text:
        if word not in stopwords.words("english") or word not in string.punctuation:
            y.append(word)

    # print(y)
    text = y.copy()
    y.clear()




    # now stem thw word
    stemmer = PorterStemmer()
    for word in text:
        y.append(stemmer.stem(word))

    return " ".join(y)


@app.route("/predict", methods=["GET","POST"])
def predict():
    pridct = None
    if request.method == "POST":
        message = request.form.get("message")


    return render_template("home.html",prediction="SPAM")



if __name__ == "__main__":
    app.run(debug=True,port=5008)
