"""
Speech Command / Keyword Spotting Dataset Loader
For rebuttal: demonstrates non-ECG TinyML domain with similar PW bottleneck
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import torchaudio
import soundfile as sf

# Note: In torchaudio 2.1+, set_audio_backend() was removed
# We use soundfile directly for audio loading to avoid torchcodec dependency

from typing import Tuple, Dict, Optional

# Standard Google Speech Commands v2 (35 classes)
# Download from: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
SPEECH_COMMANDS_CLASSES = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

# For TinyML, we typically use a subset (10-12 keywords + silence)
TINYML_KEYWORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence', 'unknown']


class SpeechCommandsDataset(Dataset):
    """
    Keyword spotting dataset for 1D audio signals.
    Returns spectrograms or raw waveforms depending on config.
    """
    def __init__(self, root: str, subset: str = 'training', 
                 use_mfcc: bool = True, n_mfcc: int = 40, 
                 sample_rate: int = 16000, max_len: int = 16000,
                 keywords: list = None, binary: bool = False):
        """
        Args:
            root: Path to speech_commands_v0.02 or similar
            subset: 'training', 'validation', or 'testing'
            use_mfcc: If True, return MFCC features; else raw waveform
            n_mfcc: Number of MFCC coefficients
            sample_rate: Expected sample rate
            max_len: Maximum length in samples (pad/truncate)
            keywords: List of keywords to use (default: TINYML_KEYWORDS)
            binary: If True, binary classification (keyword vs. unknown)
        """
        self.root = Path(root)
        self.subset = subset
        self.use_mfcc = use_mfcc
        self.n_mfcc = n_mfcc
        self.sr = sample_rate
        self.max_len = max_len
        self.keywords = keywords or TINYML_KEYWORDS
        self.binary = binary
        
        # Build file list
        self.files = []
        self.labels = []
        self._build_dataset()
        
        if use_mfcc:
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={'n_fft': 512, 'hop_length': 160, 'n_mels': 40}
            )
        
    def _build_dataset(self):
        """Scan directory and build file list"""
        # Read validation/test lists if they exist
        val_list_file = self.root / 'validation_list.txt'
        test_list_file = self.root / 'testing_list.txt'
        
        val_files = set()
        test_files = set()
        if val_list_file.exists():
            with open(val_list_file) as f:
                val_files = {line.strip() for line in f if line.strip()}
        if test_list_file.exists():
            with open(test_list_file) as f:
                test_files = {line.strip() for line in f if line.strip()}
        
        # Map subset name
        if self.subset == 'training':
            target_set = 'train'
        elif self.subset in ['validation', 'val']:
            target_set = 'val'
        else:
            target_set = 'test'
        
        # Scan directories for each keyword
        for word in self.keywords:
            if word in ['silence', 'unknown']:
                continue  # Handle separately
                
            word_dir = self.root / word
            if not word_dir.exists():
                continue
                
            for wav_file in word_dir.glob('*.wav'):
                rel_path = f"{word}/{wav_file.name}"
                
                # Determine which set this file belongs to
                if target_set == 'val' and rel_path in val_files:
                    label = self.keywords.index(word)
                    self.files.append(wav_file)
                    self.labels.append(label)
                elif target_set == 'test' and rel_path in test_files:
                    label = self.keywords.index(word)
                    self.files.append(wav_file)
                    self.labels.append(label)
                elif target_set == 'train' and rel_path not in val_files and rel_path not in test_files:
                    label = self.keywords.index(word)
                    self.files.append(wav_file)
                    self.labels.append(label)
        
        # Add "unknown" class samples from other directories
        if 'unknown' in self.keywords:
            unknown_label = self.keywords.index('unknown')
            other_dirs = [d for d in self.root.iterdir() 
                         if d.is_dir() and d.name not in self.keywords 
                         and d.name not in ['_background_noise_']]
            
            for other_dir in other_dirs[:5]:  # Limit to avoid imbalance
                for wav_file in list(other_dir.glob('*.wav'))[:100]:  # Max 100 per class
                    rel_path = f"{other_dir.name}/{wav_file.name}"
                    if target_set == 'val' and rel_path in val_files:
                        self.files.append(wav_file)
                        self.labels.append(unknown_label)
                    elif target_set == 'test' and rel_path in test_files:
                        self.files.append(wav_file)
                        self.labels.append(unknown_label)
                    elif target_set == 'train' and rel_path not in val_files and rel_path not in test_files:
                        self.files.append(wav_file)
                        self.labels.append(unknown_label)
        
        # Add silence samples (from background noise)
        if 'silence' in self.keywords:
            silence_label = self.keywords.index('silence')
            bg_dir = self.root / '_background_noise_'
            if bg_dir.exists():
                # Generate silence samples by extracting random chunks from background
                for bg_file in list(bg_dir.glob('*.wav'))[:5]:
                    # Create ~100 random chunks per background file
                    for _ in range(100):
                        self.files.append(('silence', bg_file, np.random.randint(0, 100000)))
                        self.labels.append(silence_label)
        
        print(f"[SpeechCommands] {self.subset}: loaded {len(self.files)} samples across {len(set(self.labels))} classes")
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_info = self.files[idx]
        label = self.labels[idx]
        
        # Handle silence samples specially
        if isinstance(file_info, tuple) and file_info[0] == 'silence':
            _, bg_file, offset = file_info
            # Use soundfile directly to avoid torchcodec dependency
            data, sr = sf.read(str(bg_file), dtype='float32')
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)
            # Extract random chunk
            start = min(offset, waveform.shape[1] - self.max_len)
            waveform = waveform[:, start:start + self.max_len]
        else:
            # Use soundfile directly to avoid torchcodec dependency
            data, sr = sf.read(str(file_info), dtype='float32')
            if data.ndim == 1:
                waveform = torch.from_numpy(data).unsqueeze(0)
            else:
                waveform = torch.from_numpy(data.T)
        
        # Resample if needed
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)
        
        # Pad or truncate to max_len
        if waveform.shape[1] < self.max_len:
            waveform = F.pad(waveform, (0, self.max_len - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.max_len]
        
        # Convert mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Apply MFCC if requested
        if self.use_mfcc:
            features = self.transform(waveform)  # (1, n_mfcc, time_frames)
        else:
            features = waveform  # (1, max_len)
        
        # Binary classification mode
        if self.binary:
            # Map to 0 (unknown/silence) or 1 (keyword)
            if label >= len(self.keywords) - 2:  # last two are silence/unknown
                label = 0
            else:
                label = 1
        
        return features.squeeze(0), label


def load_speech_commands_loaders(
    root: str,
    batch_size: int = 64,
    use_mfcc: bool = True,
    n_mfcc: int = 40,
    max_len: int = 16000,
    keywords: list = None,
    binary: bool = False,
    num_workers: int = 0,
    verbose: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load train/val/test dataloaders for keyword spotting.
    
    Returns:
        (train_loader, val_loader, test_loader, metadata)
    """
    keywords = keywords or TINYML_KEYWORDS
    
    # Create datasets
    train_ds = SpeechCommandsDataset(
        root, subset='training', use_mfcc=use_mfcc, n_mfcc=n_mfcc,
        max_len=max_len, keywords=keywords, binary=binary
    )
    val_ds = SpeechCommandsDataset(
        root, subset='validation', use_mfcc=use_mfcc, n_mfcc=n_mfcc,
        max_len=max_len, keywords=keywords, binary=binary
    )
    test_ds = SpeechCommandsDataset(
        root, subset='testing', use_mfcc=use_mfcc, n_mfcc=n_mfcc,
        max_len=max_len, keywords=keywords, binary=binary
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Metadata
    num_classes = 2 if binary else len(keywords)
    if use_mfcc:
        # MFCC shape: (n_mfcc, time_frames)
        # With 16kHz, 1sec audio, 512 FFT, 160 hop -> ~100 frames
        time_frames = (max_len // 160) + 1
        meta = {
            'num_channels': n_mfcc,
            'seq_len': time_frames,
            'num_classes': num_classes,
            'fs': 16000,
            'feature_type': 'mfcc'
        }
    else:
        meta = {
            'num_channels': 1,
            'seq_len': max_len,
            'num_classes': num_classes,
            'fs': 16000,
            'feature_type': 'raw_waveform'
        }
    
    if verbose:
        print(f"[SpeechCommands] Loaded {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")
        print(f"[SpeechCommands] Classes: {num_classes}, Features: {meta['num_channels']}x{meta['seq_len']}")
    
    return train_loader, val_loader, test_loader, meta


# Wrapper for experiments.py registration
def load_keyword_spotting_wrapper(batch_size=64, binary=True, **kwargs):
    """
    Simplified wrapper for keyword spotting (binary classification).
    Set SPEECH_COMMANDS_ROOT environment variable to dataset path.
    """
    root = os.environ.get('SPEECH_COMMANDS_ROOT', './data/speech_commands_v0.02')
    
    # Use MFCC with smaller parameters for TinyML
    return load_speech_commands_loaders(
        root=root,
        batch_size=batch_size,
        use_mfcc=True,
        n_mfcc=40,
        max_len=16000,
        keywords=TINYML_KEYWORDS,
        binary=binary,
        num_workers=0,
        verbose=True
    )
