#Text - Pratyush and Siri

from mmsdk import mmdatasdk as md
import pandas as pd
"""
features = {
    'CMU_MOSEI_TimestampedWords': md.cmu_mosei.highlevel['CMU_MOSEI_TimestampedWords'],
    'CMU_MOSEI_Opinion_Labels': md.cmu_mosei.highlevel['CMU_MOSEI_Opinion_Labels'],
    'CMU_MOSEI_Labels': md.cmu_mosei.highlevel['CMU_MOSEI_Labels']  # emotion labels
}

dataset = md.mmdataset(features)

data = []

for vid in dataset['CMU_MOSEI_TimestampedWords'].keys():
    text_data = dataset['CMU_MOSEI_TimestampedWords'][vid]
    sentiment_data = dataset['CMU_MOSEI_Opinion_Labels'][vid]
    emotion_data = dataset['CMU_MOSEI_Labels'][vid]

    for i, segment in enumerate(text_data['intervals']):
        start, end = segment
        words = " ".join([w[0] for w in text_data['features'][i]])  
        
        sentiment = sentiment_data['features'][i][0]
        emotions = emotion_data['features'][i]  
        
        data.append({
            'video_id': vid,
            'start': start,
            'end': end,
            'text': words,
            'sentiment': sentiment,
            'happiness': emotions[0],
            'sadness': emotions[1],
            'anger': emotions[2],
            'fear': emotions[3],
            'disgust': emotions[4],
            'surprise': emotions[5]
        })

df = pd.DataFrame
print(df.shape)
print(df.head())

"""
print(md.cmu_mosei.highlevel.keys())

dataset = md.download('CMU_MOSEI')

print(dataset.keys())