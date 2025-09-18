import os
import json
import pickle
import shutil
import datetime
import traceback
from Levenshtein import distance as lev_distance
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
# max allowed mismatches for fuzzy match
MAX_MISMATCH = 2  

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
    
# Load reference database on startup
REFERENCE_DB = os.path.join(app.config['MODELS_FOLDER'], 'reference_db.fasta')  # adjust path
reference_sequences = {}


def normalize_seq(seq):
    """Uppercase and remove whitespace/newlines for exact matching."""
    return str(seq).upper().replace("\n", "").replace("\r", "").replace(" ", "").strip()

def load_reference_db(ref_fasta):
    """Load reference DB with normalized sequences"""
    global reference_sequences
    reference_sequences = {}
    if os.path.exists(ref_fasta):
        for record in SeqIO.parse(ref_fasta, "fasta"):
            record_id = record.id.split("|")[0]
            species = record.id.split("|")[1] if "|" in record.id else "unknown_species"
            seq_norm = normalize_seq(record.seq)
            reference_sequences[seq_norm] = species
        print(f"Loaded {len(reference_sequences)} sequences from reference DB")
    else:
        print("Reference DB not found!")

load_reference_db(REFERENCE_DB)

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

'''
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
            annotation_obj = cluster_annotation.get(str(label), f"Unknown_{label}")
            if isinstance(annotation_obj, dict):
                annotation = annotation_obj.get("annotation", f"Unknown_{label}")
                action = annotation_obj.get("action", "unknown")
            else:
                annotation = annotation_obj
                action = "unknown"

            results.append({
                "sequence_id": record["id"],
                "sequence_length": record["length"],
                "cluster": int(label),
                "annotation": annotation,
                "confidence": round(float(confidence), 4),
                "action": action
            })
        
        # Calculate abundance summary
        label_counts = Counter(labels)
        abundance_summary = []
        for cluster_id, count in label_counts.items():
            annotation_obj = cluster_annotation.get(str(cluster_id), f"Unknown_{cluster_id}")
            if isinstance(annotation_obj, dict):
                annotation = annotation_obj.get("annotation", f"Unknown_{cluster_id}")
                action = annotation_obj.get("action", "unknown")
            else:
                annotation = annotation_obj
                action = "unknown"

            percentage = (count / len(sequences)) * 100
            abundance_summary.append({
                "cluster": int(cluster_id),
                "annotation": annotation,
                "count": int(count),
                "percentage": round(percentage, 2),
                "action": action
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
'''
def find_closest_species(seq, ref_dict, max_mismatch=MAX_MISMATCH):
    """Search reference DB: exact match first, then fuzzy matching"""
    seq_norm = normalize_seq(seq)
    
    # Exact match
    if seq_norm in ref_dict:
        return ref_dict[seq_norm]
    
    # Fuzzy match (only sequences of same length)
    for ref_seq, species in ref_dict.items():
        if len(ref_seq) == len(seq_norm) and lev_distance(seq_norm, ref_seq) <= max_mismatch:
            return species
    
    return None  # not found

def process_fasta(filepath):
    """Process sequences: reference DB first, then AI clustering"""
    try:
        sequences = []
        seq_records = []

        # Step 1: Read sequences from FASTA
        for record in SeqIO.parse(filepath, "fasta"):
            seq_norm = normalize_seq(record.seq)
            if len(seq_norm) >= KMER_SIZE:
                sequences.append(seq_norm)
                seq_records.append({'id': record.id, 'length': len(seq_norm)})

        if not sequences:
            return {'error': 'No valid sequences found in file'}

        results = []
        kmers_list = []
        ai_indices = []

        # Step 2: Check reference DB
        for i, seq in enumerate(sequences):
            species = find_closest_species(seq, reference_sequences)
            if species:
                results.append({
                    'sequence_id': seq_records[i]['id'],
                    'sequence_length': seq_records[i]['length'],
                    'cluster': None,
                    'annotation': species,
                    'confidence': 1.0,
                    'action': 'known'
                })
            else:
                kmers_list.append(seq_to_kmers(seq, KMER_SIZE))
                ai_indices.append(i)

        # Step 3: AI clustering for unknown sequences
        if kmers_list:
            X = vectorizer.transform(kmers_list)
            labels = kmeans.predict(X)
            confidences = compute_confidence(X, labels, kmeans)

            for idx, label, conf in zip(ai_indices, labels, confidences):
                record = seq_records[idx]
                annotation_obj = cluster_annotation.get(str(label), f"Unknown_{label}")
                if isinstance(annotation_obj, dict):
                    annotation = annotation_obj.get("annotation", f"Unknown_{label}")
                    action = annotation_obj.get("action", "unknown")
                else:
                    annotation = annotation_obj
                    action = "unknown"

                results.append({
                    'sequence_id': record["id"],
                    'sequence_length': record["length"],
                    'cluster': int(label),
                    'annotation': annotation,
                    'confidence': round(float(conf), 4),
                    'action': action
                })

        # Step 4: Abundance summary
        label_counts = Counter([r['cluster'] for r in results if r['cluster'] is not None])
        abundance_summary = []
        for cluster_id, count in label_counts.items():
            annotation_obj = cluster_annotation.get(str(cluster_id), f"Unknown_{cluster_id}")
            if isinstance(annotation_obj, dict):
                annotation = annotation_obj.get("annotation", f"Unknown_{cluster_id}")
                action = annotation_obj.get("action", "unknown")
            else:
                annotation = annotation_obj
                action = "unknown"
            percentage = (count / len(sequences)) * 100
            abundance_summary.append({
                "cluster": int(cluster_id),
                "annotation": annotation,
                "count": int(count),
                "percentage": round(percentage, 2),
                "action": action
            })
        abundance_summary.sort(key=lambda x: x['count'], reverse=True)

        # Step 5: Diversity metrics
        diversity_metrics = {
            'total_sequences': len(sequences),
            'unique_clusters': len(label_counts),
            'shannon_diversity': calculate_shannon_diversity(label_counts.values()),
            'simpson_diversity': calculate_simpson_diversity(label_counts.values()),
            'average_confidence': round(float(np.mean([r['confidence'] for r in results])), 4)
        }

        return {
            'success': True,
            'results': results,
            'abundance_summary': abundance_summary,
            'diversity_metrics': diversity_metrics,
            'timestamp': datetime.datetime.now().isoformat()
        }

    except Exception as e:
        return {'error': f'Processing error: {str(e)}'}

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
    try:
        file = request.files.get('file')
        sequence_text = request.form.get('sequence_text')

        if not file and not sequence_text:
            return jsonify({'error': 'No file or sequence text provided'}), 400

        # If a file is provided, save and process it
        if file and file.filename != '':
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file format. Only FASTA files are allowed'}), 400
            filename = secure_filename(file.filename)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            results = process_fasta(filepath)
        else:
            # If sequence text is provided, save to a temporary FASTA file
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_pasted_sequences.fasta"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'w') as f:
                f.write(sequence_text)
            results = process_fasta(filepath)

        if results.get('success'):
            output_filename = f"results_{timestamp}.json"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            results['output_file'] = output_filename

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
                'annotation': (
                    cluster_annotation.get(str(i), {}).get("annotation", f"Cluster_{i}")
                    if isinstance(cluster_annotation.get(str(i)), dict)
                    else cluster_annotation.get(str(i), f"Cluster_{i}")
                ),
                'action': (
                    cluster_annotation.get(str(i), {}).get("action", "unknown")
                    if isinstance(cluster_annotation.get(str(i)), dict)
                    else "unknown"
                )
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