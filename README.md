# Audio Classification Project

A machine learning project for audio classification using deep neural networks and Mel-Spectrogram features.

## 📋 Project Overview

This project implements an audio classification system that uses convolutional neural networks to classify audio clips into different categories (e.g., beach, bus, etc.). The model processes raw audio files and converts them into Mel-Spectrograms for feature extraction.

## 🎯 Key Features

- **Mel-Spectrogram Feature Extraction**: Transforms raw audio into visual spectrograms using torchaudio
- **Data Augmentation**: Includes techniques like time shifting, noise injection, and gain adjustment to improve model robustness
- **VGG-Inspired Architecture**: Uses stacked convolutional blocks with batch normalization for feature learning
- **Automatic Data Handling**: Loads audio files from dataset directories with metadata mapping

## 📂 Project Structure

```
├── audio_dataset.py        # Custom PyTorch Dataset class for audio loading
├── models.py              # Neural network architecture definitions
├── lab_6.py               # Main training/inference script
├── function_test.py       # Unit tests for verification
├── Lab 6.ipynb            # Jupyter notebook with experiments and visualization
├── model.pth              # Trained model weights (PyTorch)
└── README.md              # This file
```

## 🏗️ Components

### AudioClassifier Model

A convolutional neural network built with:
- **Input**: Mel-Spectrogram [1, 128, ~431] dimensions
- **Feature Extraction**: 3 VGG blocks with increasing channel depth (32→64→128)
- **Pooling**: Max pooling after each block for dimensionality reduction
- **Regularization**: Dropout (0.5) for preventing overfitting
- **Output**: Fully connected layer mapping to class labels

### LoadAudio Dataset

Custom PyTorch dataset that:
- Reads audio metadata from CSV files (filepath and label mapping)
- Loads audio files from specified directories
- Converts audio to Mel-Spectrograms with configurable parameters
- Applies data augmentation during training mode
- Handles normalization for consistent feature scaling

### Data Augmentation Techniques

- **Time Shift**: Randomly shifts audio in the time dimension
- **Noise Injection**: Adds Gaussian noise to increase robustness
- **Gain Adjustment**: Randomly adjusts volume levels

## 🚀 Getting Started

### Requirements

- Python 3.x
- PyTorch
- torchaudio
- NumPy

### Installation

```bash
pip install torch torchaudio numpy
```

### Dataset Format

Organize your audio dataset as follows:

```
dataset_root/
├── metadata.txt          # Tab-separated: filepath, label
└── audio/
    ├── audio_sample_1.wav
    ├── audio_sample_2.wav
    └── ...
```

The metadata file should contain tab-separated values:
```
audio/sample_1.wav    beach
audio/sample_2.wav    bus
```

## 💻 Usage

### Training the Model

```python
from audio_dataset import LoadAudio
from models import AudioClassifier
import torch
import torch.nn as nn
import torch.optim as optim

# Load training dataset
train_dataset = LoadAudio(
    root_dir='path/to/dataset',
    meta_filename='metadata.txt',
    audio_subdir='audio',
    training_flag=True
)

# Create model and optimizer
model = AudioClassifier(num_classes=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for audio, label in train_dataset:
        output = model(audio.unsqueeze(0))
        loss = criterion(output, label.unsqueeze(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Inference

```python
# Load trained model
model = AudioClassifier(num_classes=10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Make predictions
with torch.no_grad():
    output = model(mel_spectrogram)
    predicted_class = torch.argmax(output, dim=1)
```

## 📊 Model Architecture

```
Input (Mel-Spectrogram: [1, 128, 431])
    ↓
VGGBlock(1 → 32 channels) [MaxPool]
    ↓
VGGBlock(32 → 64 channels) [MaxPool]
    ↓
VGGBlock(64 → 128 channels) [MaxPool]
    ↓
Dropout(0.5)
    ↓
Flatten
    ↓
Fully Connected Layer
    ↓
Output (num_classes)
```

## 🔧 Configuration

Key parameters you may want to adjust:

- **sample_rate**: Audio sampling rate (default: 16000 Hz)
- **n_mels**: Number of Mel-frequency bins (default: 128)
- **n_fft**: FFT window size (default: 400)
- **num_classes**: Number of audio categories (depends on your dataset)
- **batch_size**: Training batch size
- **learning_rate**: Optimizer learning rate
- **dropout_rate**: Dropout probability (currently 0.5)

## 📈 Training Tips

1. **Data Augmentation**: Make sure to use `training_flag=True` for training dataset and `training_flag=False` for validation/test
2. **Normalization**: Mel-Spectrograms are normalized during loading
3. **Class Imbalance**: Consider using weighted loss if classes are imbalanced
4. **Learning Rate**: Start with 0.001 and adjust based on convergence

## 📝 Files Description

- **audio_dataset.py**: Implements the `LoadAudio` class for efficient audio data loading with augmentation
- **models.py**: Contains `VGGBlock` and `AudioClassifier` neural network components
- **Lab 6.ipynb**: Interactive notebook for experimentation, visualization, and testing
- **function_test.py**: Unit tests to verify implementations
- **model.pth**: Pre-trained model weights

## 🎓 Lab Assignment

This project is part of Lab 6, focusing on:
- Deep learning for audio processing
- Feature engineering (Mel-Spectrograms)
- Custom PyTorch datasets
- Model training and evaluation
- Data augmentation strategies

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [torchaudio Documentation](https://pytorch.org/audio/stable/)
- [Mel-Spectrogram Basics](https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html)

## 📄 License

This project is provided as educational material for Lab 6.

## 👤 Author

Created as a lab assignment for audio classification coursework.

---

**Last Updated**: January 2026
