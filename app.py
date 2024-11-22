from flask import Flask, request, jsonify,render_template
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('placement_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")
  

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json  # Input is expected in JSON format

    # Convert to a DataFrame
    df = pd.DataFrame([data])

    # Predict using the model
    prediction = model.predict(df)
    print(prediction[0])
    
    # Return the result
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
