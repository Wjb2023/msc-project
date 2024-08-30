import torch
import torchaudio
import torchaudio.transforms as transforms
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import numpy as np
import pandas as pd
import os

def resample_waveform(input_file, target_rate=16000):
    waveform, original_rate = torchaudio.load(input_file)
    
    # Check if resampling is needed
    if original_rate != target_rate:
        resampler = transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
        resampled_waveform = resampler(waveform)
    else:
        resampled_waveform = waveform
    print(f'New waveform size = {resampled_waveform.shape}')
    
    return resampled_waveform

def flatten_channels(waveform):
    if waveform.ndim == 2:
        waveform = torch.mean(waveform, dim=0)
    return waveform

def extract_embedding(waveform, model, feature_extractor):
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

    model.eval()  
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  
    embeddings = hidden_states[-1]

    embeddings = embeddings.squeeze() 

    return embeddings[-1, ...]

def find_mean_embedding(waveform, model, feature_extractor, segment_duration=5, overlap_duration=1):
    segment_length = segment_duration * 16000
    overlap_length = overlap_duration * 16000
    step_size = segment_length - overlap_length
    
    result_vectors = []
    
    if waveform.shape[0] < segment_length:
        return extract_embedding(waveform, model, feature_extractor)

    for start in range(0, waveform.shape[0] - segment_length + 1, step_size):
        segment = waveform[start:start + segment_length]
        
        vector = extract_embedding(segment, model, feature_extractor)
        result_vectors.append(vector)
    

    vectors_tensor = torch.stack(result_vectors)
    average_vector = torch.mean(vectors_tensor, dim=0)

    return average_vector

def main():
    csv_path = "./track_ids.csv"
    audio_dir = "../Data/wavs"

    model_name = "mtg-upf/discogs-maest-5s-pw-129e"  

    # Load the model and feature extractor
    model = AutoModelForAudioClassification.from_pretrained(model_name, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)

    id_mapping = pd.read_csv(csv_path)
    id_mapping['audio_embedding'] = None
    for idx, row in id_mapping.iterrows():
        track_id = str(row['track_7digitalid'])
        filename = track_id + '.clip.wav'
        wav_dir = os.path.join(audio_dir, filename)
        if os.path.exists(wav_dir):
            print('File found')
            waveform = resample_waveform(wav_dir)
            waveform = flatten_channels(waveform)
            embedding = find_mean_embedding(waveform, model, feature_extractor)
            print(f'Final embedding dim = {embedding.shape}')
            id_mapping.at[idx, 'audio_embedding'] = embedding.tolist()

    id_mapping.to_csv(csv_path, index=False)
    print(f'Updated CSV saved to {csv_path}') 

if __name__ == '__main__':
    main() 