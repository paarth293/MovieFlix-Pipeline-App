"""
app.py – Flask REST API for the Emotion-Based Movie Recommender
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from recommender import recommend, get_metrics, get_metadata
import os

app = Flask(__name__)
CORS(app)   # allow React dev server on :3000


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Movie Recommender API running'})


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """
    POST /api/recommend
    Body: { "mood": "Happy", "model": "svm", "top_n": 5 }
    """
    data = request.get_json(force=True)
    mood      = data.get('mood', 'Happy')
    model_key = data.get('model', 'svm').lower()
    top_n     = int(data.get('top_n', 5))

    movies, label = recommend(mood, model_key=model_key, top_n=top_n)

    if not movies:
        return jsonify({'error': label}), 400

    return jsonify({
        'mood'           : mood,
        'predicted_label': label,
        'model_used'     : model_key.upper(),
        'movies'         : movies,
    })


@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Return model comparison metrics."""
    return jsonify(get_metrics())


@app.route('/api/metadata', methods=['GET'])
def metadata():
    """Return system metadata."""
    return jsonify(get_metadata())


@app.route('/api/moods', methods=['GET'])
def moods():
    return jsonify({
        'moods': [
            {'label': 'Happy',   'emoji': '😄', 'description': 'Feel-good & uplifting films'},
            {'label': 'Sad',     'emoji': '😢', 'description': 'Emotional, thought-provoking dramas'},
            {'label': 'Angry',   'emoji': '😤', 'description': 'High-octane action & thrillers'},
            {'label': 'Relaxed', 'emoji': '😌', 'description': 'Romantic & heartwarming stories'},
            {'label': 'Neutral', 'emoji': '😐', 'description': 'Mixed genres for any occasion'},
        ]
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🎬  Movie Recommender API  →  http://localhost:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
