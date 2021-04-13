from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
Load_Model = pickle.load(open('Models/Knn_Model.pickle', 'rb'))
standScaler = pickle.load(open('Models/StandardScaler.pickle', 'rb'))


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template('index.html')


@app.route('/method', methods=['POST'])
def predict():
    Values = [float(x) for x in request.form.values()]
    Arrays = np.array(Values).reshape(1, -1)
    Scaled = standScaler.transform(Arrays)
    prediction = Load_Model.predict(Scaled)[0]
    num = [i for i in range(8)]
    klasses = ['Colombia', 'Brazil',
               'Zambia',
               'Madagascar',
               'Afghanistan',
               'Zimbabwe',
               'Ethopia',
               'Russia']
    dic = dict(zip(num, klasses))
    output = [k for i, k in dic.items() if i == prediction][0]

    return render_template("result.html", result=output)


if __name__ == "__main__":
    app.run(debug=True)
