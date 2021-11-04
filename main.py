from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from PIL import Image
from ml import FaceRecognitionEstimator

app = Flask(__name__)
cors = CORS(app)
face = FaceRecognitionEstimator()

@app.route('/', methods=['POST'])
@cross_origin()
def index():
    file = request.files['file']
    img = Image.open(file)

    return jsonify(face.predict(img))


if __name__ == "__main__":
    app.run(debug = True)      