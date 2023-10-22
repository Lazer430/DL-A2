from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import signal


# This function is used to predict the class of the image
# returns the predicted class based on highest probability
def predict_classes(model, x_test):
    y_pred = model.predict(x_test)
    classes = np.argmax(y_pred, axis=1)
    return classes


model = tf.keras.models.load_model("task1.keras")

app = Flask(__name__)


# this is the home page of our flask app and simply returns the index.html
@app.route("/")
def home():
    return render_template("index.html")


# this is the shutdown page of our flask app and kills the server
@app.get("/shutdown")
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    return "Server shut down"


# this is the predict page of our flask app and returns the predicted class of the image
# and also saves the image in the static/uploaded folder
# and returns the path of the image to be displayed on the webpage
# the image is loaded, converted to array and then expanded in dimensions
# the prediction is made and the class with highest probability is returned
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
