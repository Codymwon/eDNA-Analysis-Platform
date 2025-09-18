from Bio import Entrez, SeqIO
import json
import os
import time

Entrez.email = "manoshmekha@gmail.com"  # Replace with your real email
BATCH_SIZE = 5  # number of records to fetch at once
SLEEP_TIME = 1  # seconds to wait between requests (max ~3 requests/sec)

# Paths
json_file_path = os.path.join('outputs', 'sampled_sequences.json')
fasta_file_path = 'reference_db.fasta'

# Load JSON
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Collect record IDs
record_ids = [entry['record_id'] for entry in data]

# Map record_id → sequence
seq_dict = {entry['record_id']: entry['sequence'] for entry in data}

# Open FASTA for writing
with open(fasta_file_path, 'w') as fasta_out:

    # Process in batches
    for i in range(0, len(record_ids), BATCH_SIZE):
        batch_ids = record_ids[i:i+BATCH_SIZE]
        try:
            handle = Entrez.efetch(db="nucleotide",
                                   id=",".join(batch_ids),
                                   rettype="gb",
                                   retmode="text")
            records = SeqIO.parse(handle, "genbank")
            
            for record in records:
                rid = record.id
                species = record.annotations.get('organism', 'unknown_species')
                sequence = seq_dict.get(rid, '')
                header = f"{rid}|{species}"
                fasta_out.write(f">{header}\n")
                # Wrap sequence every 80 chars
                for j in range(0, len(sequence), 80):
                    fasta_out.write(sequence[j:j+80] + "\n")

            handle.close()
            time.sleep(SLEEP_TIME)  # respect NCBI rate limit

        except Exception as e:
            print(f"Error fetching batch {batch_ids}: {e}")

print(f"FASTA with species saved to {fasta_file_path}")
