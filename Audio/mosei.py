from mmsdk.mmdatasdk import mmdataset

dataset_path = "/Users/cynthialiu/Downloads/BWSIFILES/Capstone/MOSEI"
feature_set = {
	"CMU_MOSEI_COVAREP" : f"{dataset_path}/CMU_MOSEI_COVAREP.csd"
}

dataset = mmdataset(feature_set)

segment_id = list(dataset["CMU_MOSEI_COVAREP"].data.keys())[0]
features = dataset["CMU_MOSEI_COVAREP"].data[segment_id]["features"]

print(features[:5])



