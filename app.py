from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
try:
    with open("final_trained_crop_yield_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        # Get form data
        Crop = request.form["Crop"]
        Crop_Year = request.form["Crop_Year"]
        Season = request.form["Season"]
        State = request.form["State"]
        Area = request.form["Area"]
        Annual_Rainfall = request.form["Annual_Rainfall"]
        Fertilizer = request.form["Fertilizer"]
        Pesticide = request.form["Pesticide"]
        
        # Validate inputs
        if not all([Crop, Crop_Year, Season, State, Area, Annual_Rainfall, Fertilizer, Pesticide]):
            return render_template("predict.html", error="All fields are required")
        
        try:
            # Convert to appropriate types
            Crop_Year = int(Crop_Year)
            Area = float(Area)
            Annual_Rainfall = float(Annual_Rainfall)
            Fertilizer = float(Fertilizer)
            Pesticide = float(Pesticide)
        except ValueError:
            return render_template("predict.html", error="Invalid input values. Please check your entries.")
        
        # Prepare DataFrame for prediction
        input_df = pd.DataFrame([{
            "Crop": Crop,
            "Crop_Year": Crop_Year,
            "Season": Season,
            "State": State,
            "Area": Area,
            "Annual_Rainfall": Annual_Rainfall,
            "Fertilizer": Fertilizer,
            "Pesticide": Pesticide
        }])
        
        # Predict if model is available
        if model:
            try:
                prediction = model.predict(input_df)[0]
                prediction = round(prediction, 4)
            except Exception as e:
                print(f"Prediction error: {e}")
                return render_template("predict.html", error="Error making prediction. Please try again.")
        else:
            return render_template("predict.html", error="Prediction model not available")

    return render_template("predict.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)