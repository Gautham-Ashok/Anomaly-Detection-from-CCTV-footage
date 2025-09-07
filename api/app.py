from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import tempfile
from pathlib import Path
import sys
import logging
from werkzeug.utils import secure_filename

sys.path.append(str(Path(__file__).parent.parent))

from src.anomaly_detector import AnomalyDetector
import config.config as cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize detector
detector = AnomalyDetector()

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = cfg.MAX_UPLOAD_SIZE
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve_frontend():
    """Serve the main frontend page"""
    try:
        frontend_path = Path(__file__).parent.parent / 'frontend'
        index_file = frontend_path / 'index.html'

        if not index_file.exists():
            logger.error(f"Frontend file not found at: {index_file}")
            return jsonify({'error': 'Frontend not found', 'path': str(index_file)}), 404

        return send_from_directory(str(frontend_path), 'index.html')
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files from frontend directory"""
    try:
        frontend_path = Path(__file__).parent.parent / 'frontend'

        # Security check: prevent directory traversal
        if '..' in path or path.startswith('/'):
            return jsonify({'error': 'Invalid path'}), 400

        file_path = frontend_path / path
        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(frontend_path), path)
        else:
            return jsonify({'error': 'File not found', 'requested': path}), 404
    except Exception as e:
        logger.error(f"Error serving static file {path}: {e}")
        return jsonify({'error': str(e)}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector.model is not None,
        'categories': list(cfg.CATEGORIES.keys())
    })

@app.route('/detect', methods=['POST'])
def detect_anomaly():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Allowed: mp4, avi, mov, mkv, webm'}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            file.save(tmp_file.name)
        result = detector.detect_anomaly(tmp_file.name)
        os.unlink(tmp_file.name)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'error': 'Failed to process video', 'details': str(e)}), 500

@app.route('/analyze_url', methods=['POST'])
def analyze_video_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        result = detector.detect_anomaly(data['url'])
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        return jsonify({'error': 'Failed to process URL', 'details': str(e)}), 500

@app.route('/categories', methods=['GET'])
def get_categories():
    return jsonify({
        'categories': [{'name': name, 'id': id} for name, id in cfg.CATEGORIES.items()]
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        import psutil
        return jsonify({
            'memory_usage': f"{psutil.virtual_memory().percent}%",
            'cpu_usage': f"{psutil.cpu_percent()}%",
            'model_loaded': detector.model is not None,
            'categories_count': len(cfg.CATEGORIES),
            'status': 'online'
        })
    except ImportError:
        return jsonify({
            'status': 'psutil not available',
            'model_loaded': detector.model is not None,
            'categories_count': len(cfg.CATEGORIES)
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Could not retrieve system stats',
            'model_loaded': detector.model is not None,
            'categories_count': len(cfg.CATEGORIES)
        })

@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({
        'message': 'API is working!',
        'status': 'success',
        'model_loaded': detector.model is not None
    })

if __name__ == '__main__':
    frontend_dir = Path(__file__).parent.parent / 'frontend'
    frontend_dir.mkdir(exist_ok=True)

    logger.info("Starting Anomaly Detection Server...")
    logger.info(f"Frontend directory: {frontend_dir}")
    logger.info(f"Checking for index.html at: {frontend_dir / 'index.html'}")

    index_file = frontend_dir / 'index.html'
    if index_file.exists():
        logger.info("✓ Frontend file found!")
        logger.info(f"File size: {index_file.stat().st_size} bytes")
    else:
        logger.error("✗ Frontend file NOT found!")
        logger.error(f"Please save the HTML file to: {index_file}")

    logger.info(f"API will be available at: http://127.0.0.1:{cfg.API_PORT}")
    logger.info(f"Frontend will be available at: http://127.0.0.1:{cfg.API_PORT}")

    app.run(host=cfg.API_HOST, port=cfg.API_PORT, debug=True)
