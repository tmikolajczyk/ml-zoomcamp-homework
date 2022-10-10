import pickle
from flask import Flask, request, jsonify

with open('model2.bin','rb') as f_in:
    model = pickle.load(f_in)

with open('dv.bin','rb') as dv_in:
    dv = pickle.load(dv_in)


app = Flask('cc')
@app.route('/predict', methods=['POST'])

def predict():

    client = request.get_json()

    X = dv.transform([client])
    y_pred = round(model.predict_proba(X)[:,1][0], 3)    
    return jsonify({"Probability of getting a card": y_pred})

if __name__ == '__main__':
    app.run(debug = True, host = 'localhost', port=9696)