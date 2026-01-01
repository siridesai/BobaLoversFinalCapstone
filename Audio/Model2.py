import os
import pandas as pd
import torch
import torchaudio
from datasets import Dataset
from transformers import (
    AutoProcessor, 
    AutoModelForAudioClassification, 
    TrainingArguments, 
    Trainer
)
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
import time
import numpy as np
import evaluate


audio_root = r"C:\Users\sirid\Downloads\AudioDataset"
print("Path to dataset files:", audio_root)

def extract_emotion(filename):
    code = int(filename.split("-")[2])
    if code in (1, 2):  # Remove neutral and calm
        return None
    emotion_map = {
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fearful",
        7: "disgust",
        8: "surprised"
    }
    return emotion_map[code]

data = []
for actor_dir in sorted(os.listdir(audio_root)):
    actor_path = os.path.join(audio_root, actor_dir)
    if os.path.isdir(actor_path):
        for fname in os.listdir(actor_path):
            if fname.endswith(".wav"):
                full_path = os.path.join(actor_path, fname)
                emotion = extract_emotion(fname)
                if emotion is not None:
                    data.append({"path": full_path, "label": emotion})

df = pd.DataFrame(data)
df = df.sample(n=800, random_state=42).reset_index(drop=True)
print(df.head(10))

dataset = Dataset.from_pandas(df)
dataset = dataset.class_encode_column("label")
dataset = dataset.train_test_split(test_size=0.2, seed=42)

max_duration = 5
def preprocess(example):
    try:
        waveform, sr = torchaudio.load(example["path"])

        duration = waveform.shape[-1]/ sr
        if duration > max_duration:
            raise ValueError("Audio too long")

        waveform = waveform.mean(dim=0)  # Convert to mono
        if sr != 16000:
            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        input_values = waveform.tolist()

        if not isinstance(input_values, list):
            raise ValueError("Waveform conversion failed")

        return {"input_values": input_values, "label": example["label"]}

    except Exception as e:
        print(f" Failed to process {example['path']} — {type(e).__name__}: {e}")
        return {"input_values": None, "label": None}

print(f"Train dataset size before filtering: {len(dataset['train'])}")
train_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)
print(f"Train dataset size after map: {len(train_dataset)}")
train_dataset = train_dataset.filter(lambda example: example["input_values"] is not None)
print(f"Train dataset size after filter: {len(train_dataset)}")

test_dataset = dataset["test"].map(preprocess, remove_columns=dataset["test"].column_names)
test_dataset = test_dataset.filter(lambda example: example["input_values"] is not None)

class CustomAudioCollator:
    def __init__(self, padding_value=0.0):
        self.padding_value = padding_value

    def __call__(self, features):
        input_values = [torch.tensor(f["input_values"]) if not isinstance(f["input_values"], torch.Tensor) else f["input_values"] for f in features]
        labels = torch.tensor([f["label"] for f in features])
        padded_inputs = pad_sequence(input_values, batch_first=True, padding_value=self.padding_value)
        return {"input_values": padded_inputs, "labels": labels}

model = AutoModelForAudioClassification.from_pretrained(
    "superb/wav2vec2-base-superb-er",
    num_labels=dataset["train"].features["label"].num_classes,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./wav2vec2_ravdess",
    do_train=True,
    do_eval=True,
    logging_strategy="epoch",
    logging_dir="./logs",
    disable_tqdm=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none"
)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    if isinstance(eval_pred, tuple):
        logits, labels = eval_pred
    else:
        logits, labels = eval_pred.predictions, eval_pred.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

collator = CustomAudioCollator()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)


start_train = time.time()
train_result = trainer.train()
end_train = time.time()
train_time = end_train - start_train

print(f"\nTotal training time: {train_time:.2f} seconds")
print(f"Training accuracy: {train_result.metrics.get('train_accuracy', 'Not recorded')}")


start_test = time.time()
test_metrics = trainer.evaluate()
end_test = time.time()
test_time = end_test - start_test

print(f"Test accuracy: {test_metrics['eval_accuracy']:.4f}")
print(f"Test duration: {test_time:.2f} seconds")


model.save_pretrained("./wav2vec2_ravdess/final_model")
