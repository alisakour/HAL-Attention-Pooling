import re
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from scipy.sparse import lil_matrix, hstack
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

def clean_text(text):
    """ Cleans text by removing HTML tags and non-alphabetic characters. """
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

def load_and_prepare_data(max_vocab=10000):
    """ Loads the IMDB dataset and builds the vocabulary. """
    print("[INFO] Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    train_data = dataset['train']
    test_data = dataset['test']
    
    print(f"[INFO] Building vocabulary (Top {max_vocab} words)...")
    all_words =[]
    for item in tqdm(train_data, desc="Tokenizing Train Data"):
        all_words.extend(clean_text(item["text"]))
        
    word_counts = Counter(all_words)
    vocab =[w for w, c in word_counts.most_common(max_vocab)]
    
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = len(word2idx)
    
    return train_data, test_data, word2idx

def build_hal_matrix(train_data, word2idx, window_size=5, embed_dim=300):
    """ Constructs the HAL co-occurrence matrix and applies SVD dimensionality reduction. """
    V = len(word2idx)
    UNK_IDX = word2idx["<UNK>"]
    print(f"[INFO] Building HAL Matrix (V={V}, Window={window_size}). Skipping <UNK>...")
    
    L_matrix = lil_matrix((V, V), dtype=np.float32)
    R_matrix = lil_matrix((V, V), dtype=np.float32)
    
    for item in tqdm(train_data, desc="Computing Co-occurrences"):
        words = clean_text(item["text"])
        indices =[word2idx.get(w, UNK_IDX) for w in words]
        
        for i, target_word in enumerate(indices):
            if target_word == UNK_IDX: 
                continue
                
            start = max(0, i - window_size)
            end = min(len(indices), i + window_size + 1)
            
            for j in range(start, end):
                if i == j: continue
                context_word = indices[j]
                
                if context_word == UNK_IDX: 
                    continue
                
                distance = abs(i - j)
                weight = 1.0 / distance
                
                if j < i:
                    L_matrix[target_word, context_word] += weight
                else:
                    R_matrix[target_word, context_word] += weight
                    
    print(f"[INFO] Applying Truncated SVD to reduce dimensions to {embed_dim}...")
    hal_full = hstack([L_matrix, R_matrix])
    svd = TruncatedSVD(n_components=embed_dim, random_state=42)
    hal_reduced = svd.fit_transform(hal_full)
    
    return torch.FloatTensor(hal_reduced)

def create_dataloaders(data_split, word2idx, max_len=200, batch_size=64, shuffle=True):
    """ Tokenizes texts and constructs PyTorch DataLoaders. """
    X, Y = [],[]
    for item in tqdm(data_split, desc="Padding Sequences", leave=False):
        words = clean_text(item["text"])
        indices = [word2idx.get(w, word2idx["<UNK>"]) for w in words[:max_len]]
        indices += [word2idx["<PAD>"]] * (max_len - len(indices))
        X.append(indices)
        Y.append(item["label"])
        
    dataset = TensorDataset(torch.LongTensor(X), torch.LongTensor(Y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
