import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app= Flask(__name__)
app=app


standard_scaler=pickle.load(open('models/scaler.pkl','rb'))
lassocv_model=pickle.load(open('models/lassocv.pkl','rb'))



@app.route("/",methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
       Fueltype=float(request.form.get('Fueltype')) 
       Doornumber=float(request.form.get('Doornumber')) 
       Carlength=float(request.form.get('Carlength')) 
       Carwidth=float(request.form.get('Carwidth')) 
       Carheight=float(request.form.get('Carheight'))
       Enginesize=float(request.form.get('Enginesize')) 
       Horsepower=float(request.form.get('Horsepower')) 
       Peakrpm=float(request.form.get('Peakrpm'))  

       new_data_scaled=standard_scaler.transform([[Fueltype,Doornumber,Carlength,Carwidth,Carheight,Enginesize,Horsepower,Peakrpm]])
       result=lassocv_model.predict(new_data_scaled)

       return render_template('index.html',results=result[0])


    else:
        return render_template('index.html')
        

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
