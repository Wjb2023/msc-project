import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import os
import json

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_text_embedding(comments, tokenizer, model):
    task = 'The following is a comment left on a music video. Summarise the sentiment expressed as it relates to the piece of music'
    queries = [get_detailed_instruct(task, comment) for comment in comments]

    batch_dict = tokenizer(queries, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = torch.mean(embeddings, dim=0)

    embeddings = F.normalize(embeddings, p=2, dim=0)
    return embeddings

def main():
    csv_path = "./track_ids.csv"
    audio_dir = "../Data/wavs"
    text_dir = "../Data/comments"

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')

    id_mapping = pd.read_csv(csv_path)
    id_mapping['text_embedding'] = None

    for idx, row in id_mapping.iterrows():
        track_7digital = str(row['track_7digitalid'])
        wav_filename = track_7digital + '.clip.wav'
        wav_dir = os.path.join(audio_dir, wav_filename)
        track_id = str(row['track_id'])
        json_filename = track_id + '_comments.json'
        json_dir = os.path.join(text_dir, json_filename)
        if os.path.exists(wav_dir) and os.path.exists(json_dir):
            print('File found')
            with open(json_dir, 'r') as f:
                comments = json.load(f)
            if not comments:
                comments = ['i like this']
            if len(comments) > 30:
                comments = comments[:30]
            print(f'Embedding {len(comments)} comments')
            text_embed = get_text_embedding(comments, tokenizer, model)
            print(f'Final embedding dim = {text_embed.shape}')
            id_mapping.at[idx, 'text_embedding'] = text_embed.tolist()

    id_mapping.to_csv(csv_path, index=False)
    print(f'Updated CSV saved to {csv_path}') 

if __name__ == '__main__':
    main()