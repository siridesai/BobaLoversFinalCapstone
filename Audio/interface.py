import pickle 
import os 
from collections import defaultdict, Counter

class EmotionFingerprintDB:
    def __init__(self, db_path = "emotion_db.pkl"): ##place holder for path 
        self.db_path = db_path
        self.db = defaultdict(list)
        self.clip_metadata = {}
    
    def add_clip(self, clip_id, fingerprints, emotion):
        dominant_emotion = max(emotion_labels.items(), key = lambda x: x[1])[0]

        for fp in fingerprints:
            key = tuple(fp[:3])
            self.db[key].append((clip_id, dominant_emotion))

        self.clip_metadata[clip_id] = {
            "emotion_labels": emotion_labels,
            "dominant_emotion": dominant_emotion,
            "path": path,
            "speaker_id": speaker_id,
            "transcription": transcription,
        }

    def query(self, fingerprints): 
        vote_counter = Counter()
        for fp in fingerprints:
            key = tuple(fp[:3]) 
            if key in self.db:
                matches = self.db[key]
                for clip_id, emotion in matches: 
                    vote_counter[emotion] += 1
        return vote_counter.most_common() 

    def save(self): 
        with open(self.db_path, "wb") as f: 
            pickle.dump((self.db, self.clip_metadata), f)
    
    def load(self): 
        if os.path.exists(self.db_path): 
            with open(self.db_path, "rb") as f: 
                self.db, self.clip_metadata = pickle.load(f)
            print(f"Loaded DB with {len(self.clip_metadata)} clips.")
        else:
            print("No existing database found.")

    def list_emotions(self):
        return set(meta["emotion"] for meta in self.clip_metadata.values())
