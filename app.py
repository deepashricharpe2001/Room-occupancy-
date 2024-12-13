import pickle
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the saved model pipeline
model = pickle.load(open("models/model_pipeline.pkl", "rb"))


@app.route("/", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        # Collect form data and convert it to the correct data types
        day = int(request.form.get("date"))
        month = int(request.form.get("month"))
        year = int(request.form.get("year"))
        hotel = int(request.form.get("hotel"))
        room = request.form.get("room")
        is_weekend = int(request.form.get("holiday"))

        # Create a DataFrame that matches the model's expected input format
        new_data = pd.DataFrame(
            {
                "property_id": [hotel],
                "room_id": [room],
                "is_weekend": [is_weekend],
                "year": [year],
                "month": [month],
                "day": [day],
            }
        )

        # Predict occupancy rate using the pre-trained model pipeline

        result = round(model.predict(new_data)[0], 2)

        # Return the prediction and form data to the user
        return render_template(
            "index.html",
            result=result,
            day=day,
            month=month,
            year=year,
            hotel=hotel,
            room=room,
            is_weekend=is_weekend,
        )

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
