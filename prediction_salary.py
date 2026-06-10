import joblib
import pandas as pd
import numpy as np
from flask import Flask,render_template,request

model=joblib.load("salary_prediction.pkl")

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('model.html')

@app.route('/predict',methods=['POST'])
def salary():
    data=request.json
    df=pd.DataFrame([data])
    ans=model.predict(df)
    ans=np.expm1(ans)
    ans=f"{ans[0]:.2f}"
    return ans

if __name__=="__main__":
    app.run(debug=True)