import model1 
import EmbedQ
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

model = model1.EmotionLSTM()
model.load_state_dict(torch.load("best_model.pth"))

surprise = ('Wait, what?! I didn\'t see that coming.', "You\'re kidding me, right? That\'s insane!", "That was the last thing I expected to hear today!")
disgust = ("That\'s the grossest-looking pizza I've ever seen.", "Ew, why is there hair on this?","I can't even look at it without feeling sick.")
fear = ("What if something goes wrong?", "I don\'t know what\'s going to happen, I\'m scared.", "A chill ran down my spine when I saw the snake.")
anger = ("How could you do this?!", "This is unacceptable. I'm so done.", "I\'m going to go back down there and give them a piece of my mind.")
sadness = ("I hate my job", "I failed my test", "I\'m a terrible person")
happiness = ("I\'m so excited for the trip next week, I\'m on top of the world right now!", "This is the best day of my life!", "I just got hired for a new job!")

glove = EmbedQ.load_glove()
surprise = EmbedQ.embed_text(surprise, glove)
disgust = EmbedQ.embed_text(disgust, glove)
fear = EmbedQ.embed_text(fear, glove)
anger = EmbedQ.embed_text(anger, glove)
sadness = EmbedQ.embed_text(sadness, glove)
happiness = EmbedQ.embed_text(happiness, glove)

def pad_batch(batch):
    """
    Pads a batch of sequences to the maximum length in the batch.
    """
    sequences, emotions = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in sequences])
    emotions = torch.stack(emotions)
    return padded_sequences, lengths, emotions

surpriseDataloader = DataLoader(surprise, batch_size=32, shuffle=False, collate_fn=pad_batch)
disgustDataloader = DataLoader(disgust, batch_size=32, shuffle=False, collate_fn=pad_batch)
fearDataloader = DataLoader(fear, batch_size=32, shuffle=False, collate_fn=pad_batch)
angerDataloader = DataLoader(anger, batch_size=32, shuffle=False, collate_fn=pad_batch)
sadnessDataloader = DataLoader(sadness, batch_size=32, shuffle=False, collate_fn=pad_batch)
happinessDataloader = DataLoader(happiness, batch_size=32, shuffle=False, collate_fn=pad_batch)

model.eval()
surprise_outputs = []
with torch.no_grad():
    for inputs, lengthsSurprise, _ in surpriseDataloader:
        outputs = model(inputs, lengthsSurprise)
        surprise_outputs.append(outputs)

surprisePredictions = torch.cat(surprise_outputs, dim=0)

disgust_outputs = []
with torch.no_grad():
	for inputs, lengthsDisgust, _ in disgustDataloader:
		outputs = model(inputs, lengthsDisgust)
		disgust_outputs.append(outputs)

disgustPredictions = torch.cat(disgust_outputs, dim=0)

fear_outputs = []
with torch.no_grad():
	for inputs, lengthsDisgust, _ in fearDataloader:
		outputs = model(inputs, lengthsDisgust)
		fear_outputs.append(outputs)

fearPredictions = torch.cat(fear_outputs, dim=0)

anger_outputs = []
with torch.no_grad():
	for inputs, lengthsDisgust, _ in angerDataloader:
		outputs = model(inputs, lengthsDisgust)
		anger_outputs.append(outputs)

angerPredictions = torch.cat(anger_outputs, dim=0)

sadness_outputs = []
with torch.no_grad():
	for inputs, lengthsDisgust, _ in sadnessDataloader:
		outputs = model(inputs, lengthsDisgust)
		sadness_outputs.append(outputs)

sadnessPredictions = torch.cat(sadness_outputs, dim=0)

happiness_outputs = []
with torch.no_grad():
	for inputs, lengthsDisgust, _ in happinessDataloader:
		outputs = model(inputs, lengthsDisgust)
		happiness_outputs.append(outputs)

happinessPredictions = torch.cat(happiness_outputs, dim=0)

print("Surprise Predictions:", surprisePredictions)
print("Disgust Predictions:",disgustPredictions)
print("Fear Predictions:", fearPredictions)
print("Anger Predictions:", angerPredictions)
print("Sadness Predictions:", sadnessPredictions)
print("Happiness Predictions:", happinessPredictions)

def inputToPred(input):
	EmbedQ.load_glove()
	text = EmbedQ.embed_text(input, glove)
	textDataLoader = DataLoader(text, batch_size=32, shuffle=False, collate_fn=pad_batch)
	model.eval()
	t_outputs = []
	with torch.no_grad():
		for inputs, lengths, _ in textDataLoader:
			outputs = model(inputs, lengths)
			t_outputs.append(outputs)
	predictions = torch.cat(surprise_outputs, dim=0)
	return predictions
