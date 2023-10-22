from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import signal


# predict classes
def predict_classes(model, x_test):
    y_pred = model.predict(x_test)
    classes = np.argmax(y_pred, axis=1)
    return classes


model = tf.keras.models.load_model("task1.keras")

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
    fileUploaded = request.files["image"]
    saveImagePath = "static/uploaded/" + fileUploaded.filename
    fileUploaded.save(saveImagePath)

    image = tf.keras.preprocessing.image.load_img(
        saveImagePath, target_size=(28, 28, 3)
    )
    imageArr = tf.keras.preprocessing.image.img_to_array(image)
    imageArr = np.expand_dims(imageArr, axis=0)

    # with tf.device("/Gpu:0"):
    prediction = model.predict(imageArr)
    prediectedClass = np.argmax(prediction)

    return render_template(
        "index.html", prediction=prediectedClass, imgPath=saveImagePath
    )


# Run the Flask app
if __name__ == "__main__":
    app.run(port=5000)
