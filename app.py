from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
# Update file paths to be relative to the script
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model.pkl')
data_path = os.path.join(BASE_DIR, 'raw.csv')


app = Flask(__name__)
if os.environ.get('RENDER'):
    # Configuration for Render
    app.config.update(
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
    )


# Load the model and data
try:
    # Load raw data
    raw_df = pd.read_csv(data_path, 
                        delimiter=';',
                        encoding='cp1252',
                        low_memory=False)
    
    # Clean and process the data
    raw_df['price'] = raw_df['price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)
    raw_df['bathrooms'] = pd.to_numeric(raw_df['bathrooms'], errors='coerce')
    raw_df['bedrooms'] = pd.to_numeric(raw_df['bedrooms'], errors='coerce')
    raw_df['square_feet'] = pd.to_numeric(raw_df['square_feet'], errors='coerce')
    
    # Fill missing values
    numeric_columns = ['bathrooms', 'bedrooms', 'square_feet', 'price']
    for col in numeric_columns:
        raw_df[col] = raw_df[col].fillna(raw_df[col].median())
    
    # Clean state column
    raw_df['state'] = raw_df['state'].astype(str).str.strip()
    raw_df = raw_df[raw_df['state'].str.len() == 2]  # Only keep valid states
    
    # Calculate state price adjustments
    state_stats = raw_df.groupby('state').agg({
        'price': ['mean', 'median', 'count']
    }).round(2)
    state_stats.columns = ['mean_price', 'median_price', 'count']
    state_stats = state_stats.reset_index()
    
    # Calculate state multipliers
    median_price = state_stats['mean_price'].median()
    STATE_MULTIPLIERS = {
        state: price / median_price
        for state, price in zip(state_stats['state'], state_stats['mean_price'])
    }
    
    # Get unique states
    ALL_STATES = sorted(raw_df['state'].unique().tolist())
    
    # Load the model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    print("\nInitialization Complete:")
    print(f"Data shape: {raw_df.shape}")
    print(f"Number of states: {len(ALL_STATES)}")
    print("\nState Price Multipliers (Top 5 most expensive):")
    for state, mult in sorted(STATE_MULTIPLIERS.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{state}: {mult:.2f}x")

except Exception as e:
    print(f"Error in initialization: {e}")
    import traceback
    print(traceback.format_exc())
    model = None
    raw_df = None

def preprocess_input(data):
    try:
        # Get input values
        bedrooms = float(data['bedrooms'])
        bathrooms = float(data['bathrooms'])
        square_feet = float(data['size'])
        
        # Calculate base price per square foot from similar properties
        similar_properties = raw_df[
            (raw_df['state'] == data['state']) &
            (raw_df['bedrooms'] == bedrooms)
        ]
        avg_price_per_sqft = similar_properties['price'].mean() / similar_properties['square_feet'].mean() \
            if not similar_properties.empty else 2.0  # default value if no similar properties
        
        # Create base features
        base_features = pd.DataFrame({
            'bedrooms': [bedrooms],
            'bathrooms': [bathrooms],
            'square_feet': [square_feet]
        })

        # Create derived features
        derived_features = pd.DataFrame({
            'price_per_sqft': [avg_price_per_sqft],
            'bed_bath_ratio': [bedrooms / bathrooms],
            'total_rooms': [bedrooms + bathrooms],
            'sqft_per_room': [square_feet / (bedrooms + bathrooms)],
            'rooms_per_sqft': [(bedrooms + bathrooms) / square_feet],
            'sqft_scaled': [square_feet / 1000],
            'bedrooms_sq': [bedrooms ** 2],
            'bathrooms_sq': [bathrooms ** 2],
            'square_feet_sq': [square_feet ** 2],
            'bed_bath': [bedrooms * bathrooms],
            'bed_sqft': [bedrooms * square_feet],
            'bath_sqft': [bathrooms * square_feet],
            'bed_bath_sqft': [bedrooms * bathrooms * square_feet]
        })

        # Rest of your code remains the same...


        # 4. Create state dummies (50 states - dropping first state as reference)
        first_state = sorted(ALL_STATES)[0]
        remaining_states = [state for state in ALL_STATES if state != first_state]
        
        state_features = pd.DataFrame(0, index=[0], 
                                    columns=[f'state_{state}' for state in remaining_states])
        
        selected_state = data['state']
        if selected_state != first_state:
            state_col = f'state_{selected_state}'
            if state_col in state_features.columns:
                state_features[state_col] = 1

        # 5. Combine all features
        final_features = pd.concat([
            base_features,      # 3 features
            derived_features,   # 13 features
            state_features     # 50 features (51-1 states)
        ], axis=1)

        print("\nFeature Creation Summary:")
        print(f"Base features ({len(base_features.columns)}): {base_features.columns.tolist()}")
        print(f"Derived features ({len(derived_features.columns)}): {derived_features.columns.tolist()}")
        print(f"State features ({len(state_features.columns)}): {state_features.columns.tolist()[:5]}...")
        print(f"Total features: {len(final_features.columns)}")
        print(f"Selected state: {selected_state}")
        
        return final_features, selected_state

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/states', methods=['GET'])
def get_states():
    try:
        state_data = []
        for state in ALL_STATES:
            stats = state_stats[state_stats['state'] == state].iloc[0]
            state_data.append({
                'code': state,
                'avg_price': float(stats['mean_price']),
                'median_price': float(stats['median_price']),
                'multiplier': float(STATE_MULTIPLIERS[state]),
                'sample_count': int(stats['count'])
            })
        return jsonify(sorted(state_data, key=lambda x: x['multiplier'], reverse=True))
    except Exception as e:
        print(f"Error fetching states: {e}")
        return jsonify({'error': 'Could not fetch states'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"\nReceived prediction request: {data}")
        
        # Preprocess input
        features, selected_state = preprocess_input(data)
        
        # Make prediction
        log_prediction = model.predict(features)[0]
        
        # Convert from log scale to actual price
        base_prediction = np.exp(log_prediction)  # Convert from log scale
        
        # Apply state multiplier
        state_multiplier = STATE_MULTIPLIERS.get(data['state'], 1.0)
        final_prediction = base_prediction * state_multiplier
        
        # Get state statistics
        state_data = state_stats[state_stats['state'] == data['state']].iloc[0]
        
        # Calculate similar properties average
        similar_properties = raw_df[
            (raw_df['state'] == data['state']) &
            (raw_df['bedrooms'] == float(data['bedrooms'])) &
            (raw_df['bathrooms'] == float(data['bathrooms']))
        ]
        similar_avg = similar_properties['price'].mean() if not similar_properties.empty else None
        
        response = {
            'success': True,
            'prediction': round(final_prediction, 2),
            'state_details': {
                'state': data['state'],
                'multiplier': float(state_multiplier),
                'state_avg_price': float(state_data['mean_price']),
                'state_median_price': float(state_data['median_price']),
                'sample_count': int(state_data['count']),
                'similar_properties_avg': round(float(similar_avg), 2) if similar_avg is not None else None
            },
            'input_details': {
                'bedrooms': data['bedrooms'],
                'bathrooms': data['bathrooms'],
                'size': data['size'],
                'price_per_sqft': round(final_prediction / float(data['size']), 2)
            }
        }
        
        print("\nPrediction details:")
        print(f"Log prediction: {log_prediction}")
        print(f"Base prediction: {base_prediction}")
        print(f"Final prediction: {final_prediction}")
        print(f"State multiplier: {state_multiplier}")
        
        return jsonify(response)

    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
