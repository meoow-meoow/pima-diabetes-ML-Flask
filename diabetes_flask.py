from flask import Flask,render_template,request
import os
from pima_diabetes import run_diabetes_predictor

app = Flask(__name__)

# set development mode
app.env = 'development'

@app.route("/",methods=['GET','POST'])
def hello_world():
    if request.method == "POST":
        Pregnancies = request.form.get("Pregnancies")
        Glucose = request.form.get("Glucose")
        BloodPressure = request.form.get("BloodPressure")
        SkinThickness = request.form.get("SkinThickness")
        Insulin = request.form.get("Insulin")
        BMI = request.form.get("BMI")
        DiabetesPedigreeFunction = request.form.get("DiabetesPedigreeFunction")
        Age = request.form.get("Age")

        print(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        inputs=[float(Pregnancies),float(Glucose),float(BloodPressure),float(SkinThickness),float(Insulin),float(BMI),float(DiabetesPedigreeFunction),int(Age)]
        print(inputs)
        results=run_diabetes_predictor(inputs)
        return render_template('result.html',output=results)
    return render_template('home.html')


if __name__ == '__main__':
    # set FLASK_ENV to development
    os.environ['FLASK_ENV'] = 'development'
    app.run()