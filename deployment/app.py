from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model/svm_model.pkl', 'rb'))

@app.route("/predict", methods=["POST"])
def predict():
    MinTemp = float(request.form['MinTemp'])
    Rainfall = float(request.form['Rainfall'])
    Sunshine = float(request.form['Sunshine'])
    WindGustSpeed = float(request.form['WindGustSpeed'])
    Humidity9am = float(request.form['Humidity9am'])
    Humidity3pm = float(request.form['Humidity3pm'])
    Pressure3pm = float(request.form['Pressure3pm'])
    Cloud9am = float(request.form['Cloud9am'])
    Cloud3pm = float(request.form['Cloud3pm'])
    RainToday = float(request.form['RainToday'])
    month = float(request.form['month'])
    Location = str(request.form['Location'])
    
    Humidity = (Humidity9am+Humidity3pm)/2
    Cloud = (Cloud9am+Cloud3pm)/2

    columns = ["MinTemp", "Rainfall", "Sunshine", "WindGustSpeed", "Humidity9am",
        "Humidity3pm", "Pressure3pm", "Cloud9am", "Cloud3pm", "RainToday", "month",
        "Adelaide", "Albany", "Albury", "AliceSprings", "BadgerysCreek", "Ballarat",
        "Bendigo", "Brisbane", "Cairns", "Canberra", "Cobar", "CoffsHarbour",
        "Dartmoor", "Darwin", "GoldCoast", "Hobart", "Katherine", "Launceston",
        "Melbourne", "MelbourneAirport", "Mildura", "Moree", "MountGambier",
        "MountGinini", "Newcastle", "Nhil", "NorahHead", "NorfolkIsland", "Nuriootpa",
        "PearceRAAF", "Penrith", "Perth", "PerthAirport", "Portland", "Richmond",
        "Sale", "SalmonGums", "Sydney", "SydneyAirport", "Townsville", "Tuggeranong",
        "Uluru", "WaggaWagga", "Walpole", "Watsonia", "Williamtown", "Witchcliffe",
        "Wollongong", "Woomera"]
    columns = np.array(columns)
    loc_index = np.where(columns==Location)[0][0]
    x = np.zeros(len(columns))
    x[0] = MinTemp
    x[1] = Rainfall
    x[2] = Sunshine
    x[3] = WindGustSpeed
    x[4] = Humidity9am
    x[5] = Humidity3pm
    x[6] = Pressure3pm
    x[7] = Cloud9am
    x[8] = Cloud3pm
    x[9] = RainToday
    x[10] = month
    if loc_index >=0:
        x[loc_index] = 1
    pred = model.predict([x])[0]
    
    if pred == 1:
        output = f"Tomorrow the {Location} area will rain"
        return render_template("index.html", prediction = output, MinTemp=MinTemp, Rainfall=Rainfall, Sunshine=Sunshine, WindGustSpeed=WindGustSpeed, Humidity=Humidity, Pressure3pm=Pressure3pm, Cloud=Cloud)
    elif pred == 0:
        output = f"Tomorrow the {Location} area won't rain"
        return render_template("index.html", prediction = output, MinTemp=MinTemp, Rainfall=Rainfall, Sunshine=Sunshine, WindGustSpeed=WindGustSpeed, Humidity=Humidity, Pressure3pm=Pressure3pm, Cloud=Cloud)

@app.route("/")
def index():
    return  render_template('index.html')


    

if __name__ == "__main__":
    app.run(debug=True)