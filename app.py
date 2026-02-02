from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
from deepface import DeepFace
import requests
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
DB_NAME = os.getenv('DB_NAME', 'your_database_name')

# Initialize MongoDB client
try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print(f"Connected to MongoDB: {DB_NAME}")
except Exception as e:
    print(f"MongoDB connection error: {e}")
    db = None

# DeepFace Configuration
MODEL_NAME = "ArcFace"
DETECTOR = "retinaface"
DISTANCE_METRIC = "cosine"


def download_image(url, filename):
    """Download image from URL to temporary file"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Create temp file with proper extension
        ext = url.split('.')[-1].split('?')[0]  # Handle Cloudinary URLs with params
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}')
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        raise Exception(f"Failed to download {filename}: {str(e)}")


def verify_faces(aadhar_url, selfie_url):
    """Download images and perform face verification"""
    aadhar_path = None
    selfie_path = None
    
    try:
        aadhar_path = download_image(aadhar_url, "Aadhar")
        selfie_path = download_image(selfie_url, "Selfie")
        
        result = DeepFace.verify(
            img1_path=aadhar_path,
            img2_path=selfie_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=True
        )
        
        return {
            'verified': result['verified'],
            'distance': round(result['distance'], 4),
            'threshold': round(result['threshold'], 4),
            'model': MODEL_NAME,
            'confidence': round((1 - result['distance']) * 100, 2) if result['distance'] < 1 else 0
        }
        
    finally:
        # Cleanup temporary files
        if aadhar_path and os.path.exists(aadhar_path):
            os.remove(aadhar_path)
        if selfie_path and os.path.exists(selfie_path):
            os.remove(selfie_path)


@app.route('/api/verify-face', methods=['POST'])
def verify_face():
    """
    API endpoint to verify face matching between Aadhar and selfie
    
    Request Body:
    {
        "userId": "string (MongoDB ObjectId)"
    }
    
    Response:
    {
        "success": boolean,
        "verified": boolean,
        "distance": float,
        "threshold": float,
        "confidence": float,
        "message": string
    }
    """
    try:
        if not request.json:
            return jsonify({
                'success': False,
                'error': 'Request body must be JSON'
            }), 400
        
        user_id = request.json.get('userId')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'userId is required'
            }), 400
        
        if db is None:
            return jsonify({
                'success': False,
                'error': 'Database connection not available'
            }), 500
        
        try:
            user_object_id = ObjectId(user_id)
        except Exception:
            return jsonify({
                'success': False,
                'error': 'Invalid userId format'
            }), 400
        
        # Fetch seller data from MongoDB
        seller = db.sellers.find_one({'userId': user_object_id})
        
        if not seller:
            return jsonify({
                'success': False,
                'error': 'Seller not found for the given userId'
            }), 404
        
        # Check if both URLs exist
        aadhar_url = seller.get('aadharUrl')
        selfie_url = seller.get('selfieUrl')
        
        if not aadhar_url or not selfie_url:
            return jsonify({
                'success': False,
                'error': 'Aadhar or Selfie URL not found in database'
            }), 404
        
        # Perform verification
        verification_result = verify_faces(aadhar_url, selfie_url)
        
        # Prepare response
        response_data = {
            'success': True,
            'verified': verification_result['verified'],
            'distance': verification_result['distance'],
            'threshold': verification_result['threshold'],
            'confidence': verification_result['confidence'],
            'model': verification_result['model'],
            'message': 'Face verified - Same person' if verification_result['verified'] 
                      else 'Face verification failed - Different person'
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database': 'connected' if db is not None else 'disconnected',
        'model': MODEL_NAME
    }), 200


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'message': 'Face Verification API',
        'version': '1.0.0',
        'endpoints': {
            'POST /api/verify-face': 'Verify face matching between Aadhar and selfie',
            'GET /api/health': 'Health check endpoint',
            'GET /': 'API documentation'
        },
        'usage': {
            'endpoint': '/api/verify-face',
            'method': 'POST',
            'body': {
                'userId': 'MongoDB ObjectId of the user'
            }
        }
    }), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'

    print(f"Running on http://localhost:{port}")
    print(f"Debug mode: {debug}")
    print(f"Database: {DB_NAME}")
    print(f"Model: {MODEL_NAME}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)