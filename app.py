import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
pca = pickle.load(open('pca.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    scaler_features= scaler.transform(np.array(int_features).reshape(1, -1))
    pca_features=pca.transform(scaler_features)
    prediction = model.predict(pca_features)
    final_prediction=prediction**2
    output = round(final_prediction[0], 2)
    
    return render_template('index.html', prediction_text='DGRH should be = {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)