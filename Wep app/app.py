from flask import Flask , render_template ,request
import joblib
app = Flask(__name__)
model = joblib.load('Wep app/model.h5')
pca = joblib.load('Wep app/pca.h5')
scaler = joblib.load('Wep app/scaler.h5')

@app.route('/' , methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        inp_data = [
            request.args.get('year'),
            request.args.get('make'),
            request.args.get('body'),
            request.args.get('transmission'),
            request.args.get('state'),
            request.args.get('condition'),
            request.args.get('odometer', 0),
            request.args.get('color'),
            request.args.get('interior'),
            request.args.get('mmr', 0)
        ]
        inp_data =[int(n) for n in inp_data]
        inp_data = scaler.transform([inp_data])
        inp_data = pca.transform(inp_data)
        car_price= round(model.predict(inp_data)[0])

        return render_template('index.html' , car_price = car_price)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')