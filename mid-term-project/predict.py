import pickle
from pathlib import Path

from flask import Flask
from flask import request
from flask import jsonify

MODEL_FILE = Path(__file__).resolve().parent / "model.bin"
THRESHOLD = 0.3

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

p_index = list(model.classes_).index("p")

app = Flask("mushroom_classifier")


@app.route("/predict", methods=["POST"])
def predict():
    mushroom = request.get_json()

    if not isinstance(mushroom, dict):
        return jsonify({"error": "JSON is expected"}), 400

    X = dv.transform([mushroom])
    poisonous_probability = float(model.predict_proba(X)[0, p_index])
    poisonous = poisonous_probability >= THRESHOLD

    return jsonify({
        "poisonous_probability": poisonous_probability,
        "poisonous": poisonous,
        "class": "p" if poisonous else "e",
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)