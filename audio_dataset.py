import os
import csv
import torch
import random
from torch.utils.data import Dataset
import torchaudio  
from pathlib import Path

# -----------------------------------------------------------------------------
# IMPROVED DATASET WITH MEL-SPECTROGRAMS AND DATA AUGMENTATION
#
# Key improvements:
#   - Mel-Spectrogram features instead of raw waveforms
#   - Data augmentation during training (time shift, noise, gain)
#   - Proper normalization
#   - Consistent shape handling
#
# Returns:
#   - Mel-Spectrogram: [1, 128, ~431] (normalized)
#   - Label: integer class index
# -----------------------------------------------------------------------------


class LoadAudio(Dataset):
    def __init__(self, root_dir, meta_filename, audio_subdir, training_flag: bool = True):
        """
        Args:
            root_dir (str): Dataset root directory.
            meta_filename (str): Metadata filename inside root_dir.
            audio_subdir (str): Audio subdirectory relative to root_dir.
            training_flag (bool): When True, random transforms may be applied
                                  inside __getitem__ for data augmentation.
        """
        self.training_flag = training_flag
        
        # read metadata (tab-separated: filepath, label, clip_id)
        # 1) Store the directories/paths.
        self.root_dir = Path(root_dir)
        self.meta_path = self.root_dir / meta_filename
        self.audio_dir = self.root_dir / audio_subdir

        # 2) Scan audio_subdir for candidate files.
        self.samples = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            # 3) Read metadata: filename + label string → keep only valid files.
            for row in reader:
                if len(row) >= 2:
                    filename = row[0]   # ex: "audio/b020_90_100.wav"
                    label = row[1]      # ex: "beach"
                    filepath = self.root_dir / filename
                    if filepath.exists():
                        # 5) Store samples as list of (filepath, label_string).
                        self.samples.append((filepath, label))
        
        # 4) Construct `self.class_names` (sorted unique labels) and then
        #    `self.label_to_idx` (class_name → integer index).
        unique_labels = sorted(set(label for _, label in self.samples))

        self.num_classes = len(unique_labels)
        self.class_names = unique_labels

        self.label_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        # 5) Audio processing parameters
        self.target_length = 220500  # 5 seconds at 44.1kHz
        self.sample_rate = 44100
        
        # 6) Mel-Spectrogram transform
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,        # FFT window size
            hop_length=512,    # Step size between frames
            n_mels=128,        # Number of mel frequency bins
            f_min=20,          # Minimum frequency
            f_max=8000,        # Maximum frequency (Nyquist = 22050)
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()


    def __len__(self):
        return len(self.samples)


    def _apply_augmentation(self, waveform):
        """
        Apply LIGHT data augmentation to waveform (only during training)
        
        Augmentations:
        - Time shift: Randomly shift audio in time (reduced)
        - Gaussian noise: Add small random noise (reduced)
        - Random gain: Adjust volume (reduced)
        """
        # REDUCED time shift (±0.2 seconds instead of ±0.5)
        # Too much shifting destroys temporal patterns
        max_shift = int(0.2 * self.sample_rate)
        shift = random.randint(-max_shift, max_shift)
        waveform = torch.roll(waveform, shift, dims=1)
        
        # REDUCED Gaussian noise (SNR ~40-50 dB, very subtle)
        noise_factor = random.uniform(0.001, 0.003)
        noise = torch.randn_like(waveform) * noise_factor
        waveform = waveform + noise
        
        # REDUCED random gain (±2 dB instead of ±3 dB)
        gain_db = random.uniform(-2, 2)
        gain = 10 ** (gain_db / 20)
        waveform = waveform * gain
        
        # Clip to prevent overflow
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        return waveform

    def __getitem__(self, idx):
        # Steps to implement:
        #   1) filepath, label_str = self.samples[idx]
        filepath, label_str = self.samples[idx]

        #   2) waveform, sr = torchaudio.load(filepath)
        waveform, sr = torchaudio.load(filepath)

        #   3) If self.training_flag is True, apply augmentations here
        if self.training_flag:
          pass
        
        #   4) Ensure waveform shape is consistent (crop/pad if necessary)
        if waveform.shape[0] > 1:
          waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Ensure consistent length (crop and pad)
        current_length = waveform.shape[1]
        if current_length > self.target_length:
            # Random crop during training, center crop during evaluation
            if self.training_flag:
                start = random.randint(0, current_length - self.target_length)
                waveform = waveform[:, start:start + self.target_length]
            else:
                start = (current_length - self.target_length) // 2
                waveform = waveform[:, start:start + self.target_length]
        elif current_length < self.target_length:
            # Pad with zeros
            pad_length = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        
        # 4) Apply data augmentation (only during training)
        if self.training_flag:
            waveform = self._apply_augmentation(waveform)
        
        # 5) Convert waveform to Mel-Spectrogram
        mel_spec = self.mel_spectrogram(waveform)  # [1, n_mels, time]
        mel_spec_db = self.amplitude_to_db(mel_spec)  # Convert to decibels
        
        # 6) Normalize using GLOBAL dataset statistics (not per-sample!)
        # Using typical Mel-Spectrogram statistics for audio classification
        # This ensures consistent feature distribution across all samples
        mel_spec_db = (mel_spec_db + 80.0) / 80.0  # Normalize to roughly [0, 1] range
        # Clip to prevent outliers
        mel_spec_db = torch.clamp(mel_spec_db, -1.0, 3.0)
        
        # 7) Get label index
        label_idx = self.label_to_idx[label_str]
        
        # 8) Return (Mel-Spectrogram, label) with label as final item
        return mel_spec_db, label_idx