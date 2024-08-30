import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

from mulan_training import MuLaN, AudioSpectrogramTransformer, TextTransformer, cycle, has_duplicates

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchaudio

from beartype import beartype
    
@beartype
class MuLaNEvaluator(nn.Module):
    def __init__(
        self,
        model_dir: str,
        audio_dir: str,
        tag_csv: str,
        wav_csv: str,
        top_50_tags: list,
        batch_size: int = 16,
        hierarchical_contrastive_learning: bool = False,
        comcosida: bool = False,
        *,
        target_sample_rate: int = 22050,
    ):
        super().__init__()

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load(model_dir, hierarchical_contrastive_learning, comcosida)

        self.top_50 = top_50_tags

        self.ds = MagnaTagATuneDataset(
            audio_dir=audio_dir,
            tag_csv=tag_csv,
            wav_csv=wav_csv,
            top_50_tags=self.top_50,
            target_sample_rate=target_sample_rate,
        )

        self.ds_fields = None

        self.dl = DataLoader(self.ds, batch_size=batch_size, shuffle=False, drop_last=False)
        self.dl_iter = cycle(self.dl)

        self.generate_tag_embeddings()

        self.segment_length = 5 * target_sample_rate
        self.overlap_length = 1 * target_sample_rate
        self.step_size = self.segment_length - self.overlap_length

    def load(self, path, hierarchical_contrastive_loss=False, comcosida=False):
        path = Path(path)
        assert path.exists()
        
        audio_transformer = AudioSpectrogramTransformer(dim=256, depth=12, heads=10, attn_dropout=0.5, ff_dropout=0.5) 
        text_transformer = TextTransformer(dim=256, depth=12, heads=10, attn_dropout=0.5, ff_dropout=0.5)

        self.model = MuLaN(
            audio_transformer,
            text_transformer,
            hierarchical_contrastive_loss=hierarchical_contrastive_loss,
            comcosida=comcosida
        )

        pkg = torch.load(str(path), map_location=self._device)
        self.model.load_state_dict(pkg['model'])

        self.model.to(self._device)
        self.model.eval()

    def data_tuple_to_kwargs(self, data):
        if not self.ds_fields:
            self.ds_fields = ('wavs', 'labels')  # Updated to reflect the dataset structure
            assert not has_duplicates(self.ds_fields), 'dataset fields must not have duplicate field names'

        data_kwargs = dict(zip(self.ds_fields, data))

        return data_kwargs
    
    def generate_tag_embeddings(self):
        self.tag_embeddings = self.model.get_text_latents(raw_texts=self.top_50).to(self._device)
    
    def evaluate(self):
        all_labels = []
        all_scores = []

        with torch.no_grad():
            count = 0
            for data in self.dl:
                data_kwargs = self.data_tuple_to_kwargs(data)
                wavs = data_kwargs['wavs'].to(self._device)
                print(f'Processing batch {count}')

                # Get audio embeddings
                segment_embeddings = []
                if wavs.shape[-1] < self.segment_length:
                    audio_embeddings = self.model.get_audio_latents(wavs)
                else:
                    for start in range(0, wavs.shape[-1] - self.segment_length + 1, self.step_size):
                        segment = wavs[..., start:start + self.segment_length]
                        
                        vector = self.model.get_audio_latents(segment)
                        segment_embeddings.append(vector)
                    vectors_tensor = torch.stack(segment_embeddings)
    
                    audio_embeddings = torch.mean(vectors_tensor, dim=0)

                # Compute cosine similarity with tag embeddings
                scores = torch.nn.functional.cosine_similarity(
                    audio_embeddings.unsqueeze(1), self.tag_embeddings.unsqueeze(0), dim=-1
                )

                all_labels.append(data_kwargs['labels'].cpu().numpy())
                all_scores.append(scores.cpu().numpy())
                count += 1

        all_labels = np.concatenate(all_labels, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)

        # Calculate evaluation metrics
        auc_roc = roc_auc_score(all_labels, all_scores, average='macro')
        mAP = average_precision_score(all_labels, all_scores, average='macro')

        print(f'Zero-Shot AUC-ROC: {auc_roc:.4f}, mAP: {mAP:.4f}')
        return auc_roc, mAP
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    

class MagnaTagATuneDataset(Dataset):
    def __init__(self, tag_csv, wav_csv, audio_dir, top_50_tags, target_sample_rate=22050):
        """
        Args:
            csv_file (str): Path to the CSV file containing audio file paths and tags.
            audio_dir (str): Directory with all the audio files.
            top_50_tags (list): List of the top 50 tags to filter and use.
            target_sample_rate (int): The sample rate to which the audio should be resampled.
        """
        self.audio_dir = audio_dir
        self.tags = pd.read_csv(tag_csv, delimiter='\t')
        self.audio_data = pd.read_csv(wav_csv, delimiter='\t')
        self.top_50_tags = top_50_tags
        self.target_sample_rate = target_sample_rate
        
        tag_columns = [tag for tag in self.tags.columns if tag in top_50_tags]
        self.tags = self.tags[['clip_id'] + tag_columns]

        self.data = pd.merge(self.tags, self.audio_data, on='clip_id')
        
        self.tag_to_index = {tag: idx for idx, tag in enumerate(top_50_tags)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        clip_info = self.data.iloc[idx]
        
        audio_path = os.path.join(self.audio_dir, clip_info['mp3_path'])
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
                waveform = resampler(waveform)

            waveform = torch.squeeze(waveform)    
            
            label = torch.zeros(len(self.top_50_tags))
            for tag in self.top_50_tags:
                if clip_info[f'{tag}'] == 1:
                    label[self.tag_to_index[tag]] = 1
            
            return waveform, label
        
        except Exception as e:
            print(f"Skipping {audio_path} due to error: {e}")
            return self.__getitem__((idx + 1) % len(self))

    
def main():
    print('Starting evaluation')

    top_50_tags = ['guitar', 'classical', 'slow', 'techno', 'strings', 'drums', 'electronic', 'rock', 'fast', 'piano', 'ambient', 'beat', 'violin', 'vocal', 'synth', 'female', 'indian', 'opera', 'male', 'singing', 'vocals', 'no vocals', 'harpsichord', 'loud', 'quiet', 'flute', 'woman', 'male vocal', 'no vocal', 'pop', 'soft', 'sitar', 'solo', 'man', 'classic', 'choir', 'voice', 'new age', 'dance', 'male voice', 'female vocal', 'beats', 'harp', 'cello', 'no voice', 'weird', 'country', 'metal', 'female voice', 'choral']

    evaluator = MuLaNEvaluator(
        model_dir=model_dir,
        audio_dir=audio_dir,
        tag_csv=tag_csv,
        wav_csv=wav_csv,
        top_50_tags=top_50_tags,
        batch_size=16, 
        hierarchical_contrastive_learning=True,
        comcosida=True
    )

    print('Model ready. Starting now...')
    evaluator.evaluate()
    
model_dir = "../Results/results_exp6/mulan.epoch_100.pt"
audio_dir = "../Data/magnatune"
tag_csv = "../Data/magnatune/annotations_final.csv"
wav_csv = "../Data/magnatune/clip_info_final.csv"

if __name__ == '__main__':
    main()