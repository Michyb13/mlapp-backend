import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS



model = joblib.load('/Users/michyb/Downloads/Coding/mlapp/ml-backend/final_model1.pkl')


app = Flask( __name__ )
CORS(app)
required_fields = ['Education', 'Experience', 'Location', 'Job_Title', 'Age', 'Gender']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No data provided'}),400
   
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Missing fields: {missing_fields}'}), 400

    
    input_df = pd.DataFrame([data])  

    
    
    prediction = model.predict(input_df)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
  app.run(debug=True)

