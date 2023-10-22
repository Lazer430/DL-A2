from datetime import timedelta
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import signal
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf

# device = None
model = tf.keras.models.load_model("task2.keras")
scaler = MinMaxScaler(feature_range=(0, 1))
dateToPredict = None
dateToPredictNext = None

cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
numberOfInputDays = 7


# predict classes
def predict(model, data):
    # with tf.device(device_name=device):
    last_n_days = data[-numberOfInputDays:]
    last_n_days = last_n_days.reshape((1, numberOfInputDays, len(cols)))
    prediction = model.predict(last_n_days)
    real_predicted_price = scaler.inverse_transform(prediction)
    return real_predicted_price


def preprocessing(data):
    global dateToPredict, dateToPredictNext
    data = data.ffill()
    dateToPredict = data["Date"].tail(1).values[0]
    dateToPredict = pd.to_datetime(dateToPredict)
    # dateToPredict = dateToPredict + pd.DateOffset(days=1)
    dateToPredictNext = dateToPredict + pd.DateOffset(days=2)
    data = data.drop(["Date"], axis=1)
    if data.isna().sum().sum() != 0:
        data = data.bfill()
    data[cols] = scaler.fit_transform(data[cols])
    return data


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.get("/shutdown")
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    return "Server shut down"


@app.route("/predict", methods=["POST"])
def upload():
    fileUploaded = request.files["csv"]

    if fileUploaded != None:
        data = pd.read_csv(fileUploaded)
        last_7_days = data.tail(numberOfInputDays)
        last_7_days = preprocessing(last_7_days)
        last_7_days = np.array(last_7_days)
        prediction = predict(model, last_7_days)

        print("Date to predict: ", dateToPredict)
        print("Date to predict next: ", dateToPredictNext)

        actualData = yf.download("AAPL", start=dateToPredict, end=dateToPredictNext)

        # if "Date" in actualData.columns:
        #     actualData = actualData.Drop(["Date"], axis=1)
        actualData = str(actualData[1:].values)
        prediction = str(prediction[0])

    else:
        return render_template("index.html", prediction="No file uploaded")

    return render_template("index.html", prediction=prediction, actualData=actualData)


# Run the Flask app
if __name__ == "__main__":
    # gpus = tf.config.list_physical_devices("GPU")
    # if gpus:
    #     device = "/device:GPU:0"
    # else:
    #     device = "/device:CPU:0"
    app.run(port=5000)
