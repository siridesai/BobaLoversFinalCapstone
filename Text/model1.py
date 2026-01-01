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
import random

class EmotionLSTM(nn.Module):
        def __init__(self, input_dim = 300, hidden_dim = 128, dropout=0.3, num_layers=2):
            super(EmotionLSTM, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers = num_layers, dropout = dropout)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, 6)

        def forward(self, x, lengths):
            packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, (hidden, cell) = self.lstm(packed_input)
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            dropped = self.dropout(last_hidden)  # Apply dropout to the last hidden state
            output = self.fc(dropped)
            return output

if __name__ == "__main__":

    dataset_path = r"C:\Users\HP\BobaLoversFinalCapstoneCogworks"

    feature_set = {
        "CMU_MOSEI_Labels" : f"{dataset_path}/CMU_MOSEI_Labels.csd",
        "CMU_MOSEI_TimestampedWords" : f"{dataset_path}/CMU_MOSEI_TimestampedWords.csd",
        "CMU_MOSEI_TimestampedWordVectors" : f"{dataset_path}/CMU_MOSEI_TimestampedWordVectors.csd",
    }

    dataset = mmdatasdk.mmdataset(feature_set)

    vector_ids = dataset["CMU_MOSEI_TimestampedWordVectors"].data.keys()
    segment_ids = dataset["CMU_MOSEI_Labels"].data.keys()

    database = {}
    for vector_id in vector_ids:
        if vector_id in segment_ids:
            embedding = torch.from_numpy(np.array(dataset["CMU_MOSEI_TimestampedWordVectors"].data[vector_id]["features"], dtype=np.float32))
            label = torch.from_numpy(np.array(dataset["CMU_MOSEI_Labels"].data[vector_id]["features"][0][:6], dtype=np.float32))
            database[vector_id] = (embedding, label)

    def split_database(database):
        keys = list(database.keys())
        random.Random(42).shuffle(keys)
        
        train_db = {}
        val_db = {}
        test_db = {}

        for i, key in enumerate(keys):
            value = database[key]
            if i % 10 == 0:
                test_db[key] = value
            elif i % 5 == 0:
                val_db[key] = value
            else:
                train_db[key] = value

        return train_db, val_db, test_db

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
        
        

    train_db, val_db, test_db = split_database(database)
    train_dataset = MOSEIDataset(train_db)
    val_dataset = MOSEIDataset(val_db)
    test_dataset = MOSEIDataset(test_db)
    dataset = MOSEIDataset(database)
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_batch)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_batch)

    

    model = EmotionLSTM()
    criterion = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    def evaluate(model, dataloader, criterion):
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, lengths, labels in dataloader:
                outputs = model(inputs, lengths) 
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
        return val_loss / len(dataloader)

    #Not Finished Training Loop
    num_epochs = 10
    loss_vals = []
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        
        running_loss = 0.0
        total_batches = 0
        model.train()
        train_loss = 0.0

        for inputs, lengths, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                print(f"Batch loss: {loss.item():.4f}")
                print(f"Output shape: {outputs.shape}")
                print(f"Label shape: {labels.shape}")
                print(f"Sample output: {outputs[0]}")
                print(f"Sample label: {labels[0]}")
                
        avg_loss = running_loss / total_batches
        loss_vals.append(avg_loss)
        avg_val_loss = evaluate(model, val_dataloader, criterion)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved with validation loss:", best_val_loss)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")    
    
    print(loss_vals)
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss = evaluate(model, test_dataloader, criterion)
    print(f"Final Test Loss: {test_loss:.4f}")
