from flask import Flask, render_template, flash, request
from joblib import dump, load


app = Flask(__name__)


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/main/', methods=['GET', 'POST'])
def mainpage():
    if request.method == "POST":
        enteredPassword = request.form['password']
    else:
        return render_template('index.html')

    # Load the Vocab model
    loaded_vocab = load('title_vocab_v4.joblib')

    # Load the algorithm models
    LogisticRegression_Model = load('LogisticRegression_v2.joblib')

    passw = [enteredPassword]
    test_password = loaded_vocab.transform(passw)

    # Predict the strength
    LogisticRegression_Test = LogisticRegression_Model.predict(test_password)

    print(LogisticRegression_Test)

    label_pred = {810: 'asp.net', 4074: 'python', 6701: 'javascript', 8596: 'c++', 9303: 'git', 24182: 'c#', 26612: 'php', 27974: 'sql', 31382: 'java'}

    return render_template("main.html", LogReg=label_pred[LogisticRegression_Test[0]])

if __name__ == "__main__":
    app.run()
