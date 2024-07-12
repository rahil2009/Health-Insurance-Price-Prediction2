from flask import Flask,render_template,jsonify,request
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods =['GET','POST'])

def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')

    else:
        data = CustomData(
            age=float(request.form.get('age')),
            sex = request.form.get('sex'),
            bmi = float(request.form.get('bmi')),
            children = request.form.get('children'),
            smoker = request.form.get('smoker'),
            region = request.form.get('region')
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0],2)

        return render_template('results.html',final_result ='Your Appropriate Predicted Premium is: {}$'.format(results))
if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
