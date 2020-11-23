import numpy as np
import tensorflow as tf
from pickle import load
from flask import Flask,render_template,request,url_for
import pandas as pd
model=tf.keras.models.load_model("rainfall.h5")

lr=tf.keras.models.load_model("lr.h5")
app=Flask(__name__)

@app.route('/',methods=['POST','GET'])
def index():
    try:
        if request.method == 'POST':
            scaler_x=load(open('scaler_x.pkl', 'rb'))
            scaler_y=load(open('scaler_y.pkl', 'rb'))
            year=int(request.form['year'])
            area=float(request.form['area'])
            #print(area)
            #print(year)
            t_1=[1974.2]
            t_2=[1188.1]
            for i in range(year-2014):
                t=[]
                test=[]
                x1=t_1
                t.append(x1)
                x2=t_2
                t.append(x2)
                test.append(t)
                test=np.array(test)
                y_hat=model.predict(test)
                t_1=x2
                t_2=list(y_hat[0])
            rainfall_final=list(y_hat[0])
            #print(rainfall_final[0])
            data=[[area,rainfall_final[0]]]
            #print(pd.DataFrame(data))
            data=pd.DataFrame(data)
            data=scaler_x.transform(data.iloc[:,0:2])
            #print(data)
            res=scaler_y.inverse_transform(lr.predict(data))[0][0]
            return render_template('index.html',flag=1,production=res,rainfall=rainfall_final[0],year=year,area=area)
    except:
        return render_template('index.html',flag='num')
    return render_template('index.html',flag=0)
if __name__ == "__main__":
    app.run(debug=True,port=5000)
    
