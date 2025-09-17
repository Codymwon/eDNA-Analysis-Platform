import os
import json
import pickle
import shutil
import datetime
import traceback
from pathlib import Path
from collections import Counter
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from Bio import SeqIO
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MODELS_FOLDER'] = 'models'  # Adjust path to your models
ALLOWED_EXTENSIONS = {'fasta', 'fa', 'fna'}

# K-mer configuration
KMER_SIZE = 6

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load models on startup
vectorizer = None
kmeans = None
cluster_annotation = None

def load_models():
    """Load the trained models"""
    global vectorizer, kmeans, cluster_annotation
    try:
        models_dir = app.config['MODELS_FOLDER']
        vectorizer = pickle.load(open(os.path.join(models_dir, "vectorizer.pkl"), "rb"))
        kmeans = pickle.load(open(os.path.join(models_dir, "cluster_model.pkl"), "rb"))
        with open(os.path.join(models_dir, "cluster_annotation.json"), "r") as f:
            cluster_annotation = json.load(f)
        print("Models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def seq_to_kmers(seq, k=6):
    """Convert sequence to space-separated k-mers, ignoring those with 'N'"""
    return " ".join([seq[i:i+k] for i in range(len(seq)-k+1) if "N" not in seq[i:i+k]])

def compute_confidence(X, labels, kmeans_model):
    """Compute normalized confidence based on distance to cluster centroids"""
    distances = np.zeros(len(labels))
    for i, label in enumerate(labels):
        distances[i] = np.linalg.norm(X[i].toarray() - kmeans_model.cluster_centers_[label])
    max_distance = np.max(distances) if np.max(distances) > 0 else 1
    confidences = 1 - (distances / max_distance)
    return confidences

def process_fasta(filepath):
    """Process FASTA file and return classification results"""
    try:
        # Read sequences
        sequences = []
        seq_records = []
        
        for record in SeqIO.parse(filepath, "fasta"):
            seq = str(record.seq).upper()
            if len(seq) >= KMER_SIZE:
                sequences.append(seq)
                seq_records.append({
                    'id': record.id,
                    'description': record.description,
                    'length': len(seq)
                })
        
        if not sequences:
            return {'error': 'No valid sequences found in file'}
        
        # Convert to k-mers
        kmers_list = [seq_to_kmers(seq, KMER_SIZE) for seq in sequences]
        
        # Vectorize
        X = vectorizer.transform(kmers_list)
        
        # Predict clusters
        labels = kmeans.predict(X)
        
        # Compute confidence scores
        confidences = compute_confidence(X, labels, kmeans)
        
        # Prepare results
        results = []
        for i, (record, label, confidence) in enumerate(zip(seq_records, labels, confidences)):
            annotation = cluster_annotation.get(str(label), f"Unknown_{label}")
            results.append({
                'sequence_id': record['id'],
                'sequence_length': record['length'],
                'cluster': int(label),
                'annotation': annotation,
                'confidence': round(float(confidence), 4),
                'action': cluster_annotation.get(str(label), {}).get('action', 'unknown') if isinstance(cluster_annotation.get(str(label)), dict) else 'unknown'
            })
        
        # Calculate abundance summary
        label_counts = Counter(labels)
        abundance_summary = []
        for cluster_id, count in label_counts.items():
            annotation = cluster_annotation.get(str(cluster_id), f"Unknown_{cluster_id}")
            percentage = (count / len(sequences)) * 100
            abundance_summary.append({
                'cluster': int(cluster_id),
                'annotation': annotation,
                'count': int(count),
                'percentage': round(percentage, 2),
                'action': cluster_annotation.get(str(cluster_id), {}).get('action', 'unknown') if isinstance(cluster_annotation.get(str(cluster_id)), dict) else 'unknown'
            })
        
        # Sort abundance summary by count
        abundance_summary.sort(key=lambda x: x['count'], reverse=True)
        
        # Calculate diversity metrics
        diversity_metrics = {
            'total_sequences': len(sequences),
            'unique_clusters': len(label_counts),
            'shannon_diversity': calculate_shannon_diversity(label_counts.values()),
            'simpson_diversity': calculate_simpson_diversity(label_counts.values()),
            'average_confidence': round(float(np.mean(confidences)), 4)
        }
        
        return {
            'success': True,
            'results': results,
            'abundance_summary': abundance_summary,
            'diversity_metrics': diversity_metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        return {'error': f'Processing error: {str(e)}', 'traceback': traceback.format_exc()}

def calculate_shannon_diversity(counts):
    """Calculate Shannon diversity index"""
    counts = list(counts)
    total = sum(counts)
    if total == 0:
        return 0
    proportions = [c/total for c in counts]
    shannon = -sum(p * np.log(p) for p in proportions if p > 0)
    return round(float(shannon), 4)

def calculate_simpson_diversity(counts):
    """Calculate Simpson diversity index"""
    counts = list(counts)
    total = sum(counts)
    if total <= 1:
        return 0
    simpson = sum(c * (c - 1) for c in counts) / (total * (total - 1))
    return round(1 - simpson, 4)

@app.route('/')
def index():
    """Serve the main page"""
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': vectorizer is not None and kmeans is not None,
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Only FASTA files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the file
        results = process_fasta(filepath)
        
        # Save results to file
        if results.get('success'):
            output_filename = f"results_{timestamp}.json"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            results['output_file'] = output_filename
        
        # Clean up uploaded file (optional)
        # os.remove(filepath)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/download/<filename>', methods=['GET'])
def download_results(filename):
    """Download results file"""
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True, download_name=filename)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get information about loaded models"""
    if kmeans is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    return jsonify({
        'n_clusters': kmeans.n_clusters,
        'n_features': vectorizer.max_features if hasattr(vectorizer, 'max_features') else 'Unknown',
        'kmer_size': KMER_SIZE,
        'clusters': [
            {
                'id': i,
                'annotation': cluster_annotation.get(str(i), f"Cluster_{i}"),
                'action': cluster_annotation.get(str(i), {}).get('action', 'unknown') if isinstance(cluster_annotation.get(str(i)), dict) else 'unknown'
            }
            for i in range(kmeans.n_clusters)
        ]
    })

@app.route('/api/update-annotation', methods=['POST'])
def update_annotation():
    """Update cluster annotations"""
    try:
        data = request.json
        cluster_id = str(data.get('cluster_id'))
        new_annotation = data.get('annotation')
        new_action = data.get('action', 'unknown')
        
        if cluster_id in cluster_annotation:
            if isinstance(cluster_annotation[cluster_id], dict):
                cluster_annotation[cluster_id]['annotation'] = new_annotation
                cluster_annotation[cluster_id]['action'] = new_action
            else:
                cluster_annotation[cluster_id] = {
                    'annotation': new_annotation,
                    'action': new_action
                }
            
            # Save updated annotations
            annotation_path = os.path.join(app.config['MODELS_FOLDER'], "cluster_annotation.json")
            with open(annotation_path, 'w') as f:
                json.dump(cluster_annotation, f, indent=2)
            
            return jsonify({'success': True, 'message': 'Annotation updated successfully'})
        else:
            return jsonify({'error': 'Invalid cluster ID'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        print("Server starting with models loaded...")
        app.run(debug=True, port=5000)
    else:
        print("Warning: Server starting without models. Please check model files.")
        app.run(debug=True, port=5000)