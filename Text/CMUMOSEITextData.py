import pandas as pd
import pickle
import torch
import os
from mmsdk import mmdatasdk 

feature_set = {
	"CMU_MOSEI_Labels" : f"{dataset_path}/CMU_MOSEI_Labels.csd",
    "CMU_MOSEI_TimestampedWords" : f"{dataset_path}/CMU_MOSEI_TimestampedWords.csd",
    "CMU_MOSEI_TimestampedWordVectors" : f"{dataset_path}/CMU_MOSEI_TimestampedWordVectors.csd",
}

dataset = mmdatasdk.mmdataset(feature_set)
id_to_label = None
id_to_vector = None

#make caches
cache_dir = ".cache"
os.makedirs(cache_dir, exist_ok=True)
labels_cache = os.path.join(cache_dir, "mosei_labels.pkl")
vectors_cache = os.path.join(cache_dir, "mosei_word_vectors.pkl")

def _load_labels(self, dataset_path):
	if id_to_label is not None:
		return id_to_label
	
	if os.path.exists(labels_cache):
		with open(labels_cache, "rb") as f:
			print("labels loading form cache")
			id_to_label = pickle.load(f)
	else:
		print("labels loading, will save to cache")
		dataset = mmdatasdk.mmdataset({"CMU_MOSEI_Labels": f"{dataset_path}/CMU_MOSEI_Labels.csd"})
		segment_ids = list(dataset["CMU_MOSEI_Labels"].data.keys())
		id_to_label = {}
		for segment_id in segment_ids:
			id_to_label[segment_id] = dataset["CMU_MOSEI_Labels"].data[segment_id]["features"][0]
		with open(labels_cache, "wb") as f:
			pickle.dump(id_to_label, f)
		return id_to_label
	
def _load_word_vectors(self, dataset_path):
	if id_to_vector is not None:
		return id_to_vector

	if os.path.exists(vectors_cache):
		with open(vectors_cache, "rb") as f:
			print("vectors loading from cache")
			id_to_vector = pickle.load(f)
	else:
		print("vecetors loading, will save to cache")
		dataset = mmdatasdk.mmdataset({"CMU_MOSEI_Labels": f"{dataset_path}/CMU_MOSEI_Labels.csd", "CMU_MOSEI_WordVectors": f"{dataset_path}CMU_MOSEI_TimestampedWordVectors"})
		segment_ids = list(dataset["CMU_MOSEI_Labels"].data.keys())
		id_to_vector = {}
		for segment_id in segment_ids:
			features = dataset["CMU_MOSEI_TimestampedWordVectors"].data[segment_id]["features"]
			np_vectors = features[:]
			id_to_vector[segment_id] = torch.tensor(np_vectors)
		with open(vectors_cache, "wb") as f:
			pickle.dump(id_to_vector, f)
		return id_to_vector

def get_text(self, id):
	return self.id_to_text.get(id, None)

def get_label(self, id):
	return self.id_to_label.get(id, None)
	
def get_ids_by_label(self, label):
	return list(self.label_to_ids.get(label, []))
	