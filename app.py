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
    return render_template("index.html")

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
    y.clear()                   # clear privious list
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
    message = None
    if request.method == "POST":
        message = request.form.get("message")
    message = transform_text(message)
    print(message)
    
    vector = vectorizer.transform([message]).toarray()
    print(vector)
    ans = model.predict(vector)
    if ans == 0:
        predict = "Not SPAM"
    else:
        predict = "SPAM"
    print(predict)

    return render_template("index.html",prediction=predict)



if __name__ == "__main__":
    app.run(debug=True)
