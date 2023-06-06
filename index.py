from flask import Flask,render_template,request
import pickle
import numpy as np

model = pickle.load(open('model','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def agriculture():
    n = int(request.form.get('n'))
    p = int(request.form.get('p'))
    k = int(request.form.get('k'))
    t = int(request.form.get('t'))
    h = int(request.form.get('h'))
    ph =int(request.form.get('ph'))
    r = int(request.form.get('r')) 
    result = model.predict(np.array([n,p,k,t,h,ph,r]).reshape(1,7))
    
    return render_template('app.html',result=result[0])
    

    


if __name__ == '__main__':
    app.run(debug=True)


