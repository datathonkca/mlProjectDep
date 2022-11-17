from flask import Flask, render_template
import numpy as np
app=Flask(__name__)

@app.route('/') 

def home():
    return render_template('index.html')
    
@app.route('/predict',methods=['POST'])
def predict():
    from flask import request
    features_models=[i for i in request.form.values()]
    date_entry=features_models[-1]
    features_models=[features_models[0],features_models[1],features_models[2],features_models[3]]
    features_models=[float(j) for j in features_models]
    features_models=np.array([features_models]).reshape(1,4)
    import joblib
    model=joblib.load('RFR.ml')
    prediction=np.expm1(model.predict(features_models)[0])
    import pandas as pd
    date_entry=pd.to_datetime(date_entry)
    from datetime import datetime, timedelta
    date_discharged=date_entry + timedelta(days=prediction)
    week_day= date_discharged.day_name()
    month= date_discharged.month_name()
    day= date_discharged.day
    year= date_discharged.year
    text_res='This patient will be discharged on' +str(week_day) +","+str(day)+ str(month) +str(year)
    return render_template('index.html', prediction_res=' {}'.format(text_res))
    
if __name__=="__main__":
    app.run(debug=True)
    
