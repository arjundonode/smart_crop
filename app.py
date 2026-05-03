from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings

app = Flask(__name__)

# ================= LOAD MODEL & SCALER =================
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

# ================= LOAD DATASET =================
try:
    data = pd.read_csv("crop_data.csv")
    print(f"✅ Dataset loaded: {len(data)} rows")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    data = None

# ================= GLOBAL SENSOR DATA =================
latest_sensor_data = {
    "soil_moisture": 0,
    "raw_value": 0,
    "timestamp": None
}

# ================= DEFAULT VALUES =================
default_user_values = {
    "N": 0,
    "P": 0,
    "K": 0,
    "ph": 7.0,
    "temperature": 25.0,
    "humidity": 60.0,
    "rainfall": 100.0
}

# ================= CROP WATER REQUIREMENTS =================
crop_water_requirements = {
    "rice": 80, "maize": 60, "chickpea": 40, "kidneybeans": 50,
    "pigeonpeas": 45, "mothbeans": 35, "mungbean": 45, "blackgram": 45,
    "lentil": 40, "pomegranate": 50, "banana": 70, "mango": 50,
    "grapes": 45, "watermelon": 60, "muskmelon": 55, "apple": 50,
    "orange": 50, "papaya": 60, "coconut": 65, "cotton": 45,
    "jute": 70, "coffee": 65
}

def get_water_suitability(crop, soil_moisture):
    """Check if soil moisture is suitable for the crop"""
    if crop in crop_water_requirements:
        required = crop_water_requirements[crop]
        if soil_moisture >= required:
            return "✅ Suitable", "Water condition is good"
        elif soil_moisture >= required - 20:
            return "⚠️ Moderate", f"Needs {required - soil_moisture}% more moisture"
        else:
            return "❌ Unsuitable", f"Requires {required}% moisture (currently {soil_moisture}%)"
    else:
        return "⚠️ Unknown", "Water requirement not in database"

def get_combined_data(user_input):
    """Combine user input for prediction"""
    combined = {
        "N": user_input["N"],
        "P": user_input["P"],
        "K": user_input["K"],
        "temperature": user_input["temperature"],
        "humidity": user_input["humidity"],
        "ph": user_input["ph"],
        "rainfall": user_input["rainfall"]
    }
    return combined

def predict_crop(data_input):
    """Predict crops based on input parameters"""
    if model is None or scaler is None:
        return ["No model loaded"], [0]
    
    try:
        features = np.array([[
            data_input["N"],
            data_input["P"],
            data_input["K"],
            data_input["temperature"],
            data_input["humidity"],
            data_input["ph"],
            data_input["rainfall"]
        ]])
        
        # Apply scaling
        features = scaler.transform(features)
        
        probs = model.predict_proba(features)[0]
        classes = model.classes_
        
        top_indices = np.argsort(probs)[-8:][::-1]
        
        crops = [classes[i] for i in top_indices]
        scores = [round(probs[i] * 100, 2) for i in top_indices]
        
        return crops, scores
    except Exception as e:
        print(f"Prediction error: {e}")
        return ["Error in prediction"], [0]

def suggest_improvements(crop, current):
    """Suggest improvements for a specific crop"""
    if data is None:
        return {}, {}
    
    try:
        crop_data = data[data['label'] == crop]
        
        if crop_data.empty:
            return {}, {}
        
        avg = crop_data[['N','P','K','temperature','humidity','ph','rainfall']].mean()
        
        suggestions = {}
        fertilizer = {}
        
        factors = {
            "N": {"fert": "Urea", "ratio": 2.17},
            "P": {"fert": "DAP", "ratio": 2.29},
            "K": {"fert": "MOP", "ratio": 1.67}
        }
        
        for key in avg.index:
            diff = round(avg[key] - current[key], 2)
            
            if abs(diff) < 1:
                suggestions[key] = "Optimal"
            elif diff > 0:
                suggestions[key] = f"Increase by {diff}"
            else:
                suggestions[key] = f"Decrease by {abs(diff)}"
            
            # Fertilizer calculation
            if key in factors and diff > 0:
                fert_name = factors[key]["fert"]
                ratio = factors[key]["ratio"]
                kg_per_acre = round(diff * ratio, 2)
                fertilizer[key] = f"{fert_name}: {kg_per_acre} kg/acre"
        
        return suggestions, fertilizer
    except Exception as e:
        print(f"Suggestions error: {e}")
        return {}, {}

@app.route('/', methods=['GET', 'POST'])
def home():
    try:
        if request.method == 'POST':
            # Get user input
            user_input = {
                "N": float(request.form.get('N', 0)),
                "P": float(request.form.get('P', 0)),
                "K": float(request.form.get('K', 0)),
                "ph": float(request.form.get('ph', 7.0)),
                "temperature": float(request.form.get('temperature', 25.0)),
                "humidity": float(request.form.get('humidity', 60.0)),
                "rainfall": float(request.form.get('rainfall', 100.0))
            }
            
            # Make prediction
            combined_data = get_combined_data(user_input)
            crops, scores = predict_crop(combined_data)
            
            # Filter crops based on soil moisture
            soil_moisture = latest_sensor_data["soil_moisture"]
            suitable_crops = []
            suitable_scores = []
            water_status = []
            
            for crop, score in zip(crops, scores):
                suitability, message = get_water_suitability(crop, soil_moisture)
                suitable_crops.append(crop)
                suitable_scores.append(score)
                water_status.append({"suitability": suitability, "message": message})
            
            # Get suggestions if a crop was selected
            selected_crop = request.form.get('selected_crop')
            suggestions = None
            fertilizer = None
            water_advice = None
            
            if selected_crop:
                suggestions, fertilizer = suggest_improvements(selected_crop, combined_data)
                suitability, water_advice = get_water_suitability(selected_crop, soil_moisture)
            
            return render_template(
                'index.html',
                user_data=user_input,
                sensor_data=latest_sensor_data,
                crops=suitable_crops,
                scores=suitable_scores,
                water_status=water_status,
                suggestions=suggestions,
                fertilizer=fertilizer,
                water_advice=water_advice,
                selected_crop=selected_crop,
                error=None
            )
        
        # GET request - show default values
        combined_data = get_combined_data(default_user_values)
        crops, scores = predict_crop(combined_data)
        
        soil_moisture = latest_sensor_data["soil_moisture"]
        water_status = []
        for crop in crops:
            suitability, message = get_water_suitability(crop, soil_moisture)
            water_status.append({"suitability": suitability, "message": message})
        
        return render_template(
            'index.html',
            user_data=default_user_values,
            sensor_data=latest_sensor_data,
            crops=crops,
            scores=scores,
            water_status=water_status,
            suggestions=None,
            fertilizer=None,
            water_advice=None,
            selected_crop=None,
            error=None
        )
    except Exception as e:
        print(f"Home route error: {e}")
        return render_template(
            'index.html',
            user_data=default_user_values,
            sensor_data=latest_sensor_data,
            crops=[],
            scores=[],
            water_status=[],
            suggestions=None,
            fertilizer=None,
            water_advice=None,
            selected_crop=None,
            error=f"An error occurred: {str(e)}"
        )

@app.route('/sensor-data', methods=['POST'])
def sensor_data():
    global latest_sensor_data
    from datetime import datetime
    
    try:
        if request.is_json:
            sensor_json = request.json
            latest_sensor_data["soil_moisture"] = sensor_json.get("soil_moisture", 0)
            latest_sensor_data["raw_value"] = sensor_json.get("raw_value", 0)
        else:
            latest_sensor_data["soil_moisture"] = float(request.form.get('soil_moisture', 0))
            latest_sensor_data["raw_value"] = float(request.form.get('raw_value', 0))
        
        latest_sensor_data["timestamp"] = datetime.now().isoformat()
        
        print(f"\n📊 Soil moisture updated: {latest_sensor_data['soil_moisture']}%")
        
        return jsonify({
            "status": "success", 
            "message": "Soil moisture data received",
            "soil_moisture": latest_sensor_data["soil_moisture"]
        })
    except Exception as e:
        print(f"Error in sensor-data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/get-soil-moisture')
def get_soil_moisture():
    return jsonify({
        "soil_moisture": latest_sensor_data["soil_moisture"],
        "raw_value": latest_sensor_data["raw_value"],
        "timestamp": latest_sensor_data["timestamp"]
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            user_input = request.json
        else:
            user_input = {
                "N": float(request.form.get('N', 0)),
                "P": float(request.form.get('P', 0)),
                "K": float(request.form.get('K', 0)),
                "ph": float(request.form.get('ph', 7.0)),
                "temperature": float(request.form.get('temperature', 25.0)),
                "humidity": float(request.form.get('humidity', 60.0)),
                "rainfall": float(request.form.get('rainfall', 100.0))
            }
        
        combined_data = get_combined_data(user_input)
        crops, scores = predict_crop(combined_data)
        
        soil_moisture = latest_sensor_data["soil_moisture"]
        water_advice = {}
        for crop in crops:
            suitability, message = get_water_suitability(crop, soil_moisture)
            water_advice[crop] = {"suitability": suitability, "message": message}
        
        return jsonify({
            "success": True,
            "crops": crops,
            "scores": scores,
            "water_advice": water_advice,
            "input_data": user_input,
            "soil_moisture": soil_moisture
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

@app.route('/suggestions', methods=['POST'])
def get_suggestions():
    try:
        data_input = request.json
        crop = data_input.get('crop')
        
        soil_data = {
            "N": float(data_input.get('N', 0)),
            "P": float(data_input.get('P', 0)),
            "K": float(data_input.get('K', 0)),
            "temperature": float(data_input.get('temperature', 25.0)),
            "humidity": float(data_input.get('humidity', 60.0)),
            "ph": float(data_input.get('ph', 7.0)),
            "rainfall": float(data_input.get('rainfall', 100.0))
        }
        
        suggestions, fertilizer = suggest_improvements(crop, soil_data)
        
        soil_moisture = latest_sensor_data["soil_moisture"]
        suitability, water_message = get_water_suitability(crop, soil_moisture)
        
        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "fertilizer": fertilizer,
            "water_advice": {
                "suitability": suitability,
                "message": water_message,
                "current_moisture": soil_moisture
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🌾 Crop Recommendation System with Soil Moisture")
    print("="*50)
    print(f"📍 Server running on: http://10.2.0.2:5000")
    print(f"📡 Waiting for ESP32 sensor data on: /sensor-data")
    print("="*50 + "\n")
    
    # CRITICAL: host='0.0.0.0' makes it accessible to ESP32 on the network
    app.run(debug=True, host='0.0.0.0', port=5000)