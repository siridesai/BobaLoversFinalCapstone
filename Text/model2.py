import pandas as pd
from mmsdk import mmdatasdk
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim import Adam
from torch import save
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch

dataset_path = r"C:\Users\HP\BobaLoversFinalCapstoneCogworks"

feature_set = {
	"CMU_MOSEI_Labels" : "CMU_MOSEI_Labels.csd",
    "CMU_MOSEI_TimestampedWords" : "CMU_MOSEI_TimestampedWords.csd",
    "CMU_MOSEI_TimestampedWordVectors" : "CMU_MOSEI_TimestampedWordVectors.csd",
}

dataset = mmdatasdk.mmdataset(feature_set)

vector_ids = dataset["CMU_MOSEI_TimestampedWordVectors"].data.keys()
segment_ids = dataset["CMU_MOSEI_Labels"].data.keys()

database = {}
for vector_id in vector_ids:
    if vector_id in segment_ids:
        embedding = torch.tensor(dataset["CMU_MOSEI_TimestampedWordVectors"].data[vector_id]["features"], dtype=torch.float32)
        label = torch.tensor(dataset["CMU_MOSEI_Labels"].data[vector_id]["features"][0][:6], dtype=torch.float32)
        database[vector_id] = (embedding, label)

def pad_batch(batch):
    """
    Pads a batch of sequences to the maximum length in the batch.
    """
    sequences, emotions = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in sequences])
    emotions = torch.stack(emotions)
    return padded_sequences, lengths, emotions

class MOSEIDataset(Dataset):
    def __init__(self, database):
        self.ids = list()
        self.ids = list(database.keys())
        self.embeddings = list(x[0] for x in database.values())
        self.labels = list(x[1] for x in database.values())

    def __len__(self): # Added the __len__ method
        return len(self.ids)

    def __getitem__(self, id):
        embedding = self.embeddings[id]
        label = self.labels[id]
        return embedding, label

dataset = MOSEIDataset(database)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_batch)

class EmotionLSTM(nn.Module):
    def __init__(self, input_dim = 300, hidden_dim = 128):
        super(EmotionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 6)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        output = self.fc(hidden[-1])
        return output

model = EmotionLSTM()
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

#Not Finished Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, lengths, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save the model
save(model.state_dict(), "emotion_lstm_model.pth")