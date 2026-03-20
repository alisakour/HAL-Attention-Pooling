import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import random
import os
from tqdm.auto import tqdm

from data_utils import load_and_prepare_data, build_hal_matrix, create_dataloaders
from models import BaselineHALMeanPooling, RobustHALAttention

sns.set_theme(style='whitegrid', context='paper', font_scale=1.2)

def set_seed(seed=42):
    """ Ensures complete reproducibility across different runs. """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'[INFO] Random seed globally set to {seed}')

def train_and_evaluate(model, optimizer, criterion, train_loader, test_loader, word2idx, device, epochs=30, patience=5, model_name='Model'):
    print(f'\n[INFO] Commencing training phase for: {model_name}')
    history = {'train_loss':[], 'train_acc': [], 'test_acc':[]}
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1:02d}/{epochs} [Train]', leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            mask = (batch_x == word2idx['<PAD>'])
            
            optimizer.zero_grad()
            output = model(batch_x, mask)
            logits = output[0] if isinstance(output, tuple) else output
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        avg_loss = total_loss / len(train_loader)
        train_acc = (correct_train / total_train) * 100
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        
        # --- Evaluation Phase ---
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                mask = (batch_x == word2idx['<PAD>'])
                output = model(batch_x, mask)
                logits = output[0] if isinstance(output, tuple) else output
                
                preds = torch.argmax(logits, dim=1)
                correct_test += (preds == batch_y).sum().item()
                total_test += batch_y.size(0)
                
        test_acc = (correct_test / total_test) * 100
        history['test_acc'].append(test_acc)
        
        print(f'   Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')
        
        # --- Early Stopping & Checkpointing ---
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f'      [SAVE] New optimal weights saved (Test Acc: {best_acc:.2f}%)')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'      [STOP] Early stopping triggered. No validation improvement for {patience} consecutive epochs.')
                break
            
    # Restore the best performing model weights before returning
    model.load_state_dict(best_wts)
    print(f'[INFO] Training completed for {model_name}. Successfully restored the best model with Test Acc: {best_acc:.2f}%\n')
    return model, history

def plot_results(history_base, history_attn):
    print('[INFO] Generating and exporting comparative plots...')
    
    # 1. Test Accuracy Convergence
    plt.figure(figsize=(8, 5.5))
    plt.plot(range(1, len(history_attn['test_acc']) + 1), history_attn['test_acc'], marker='o', color='#8b5cf6', linewidth=2.5, markersize=8, label='HAL + Attention (Proposed)')
    plt.plot(range(1, len(history_base['test_acc']) + 1), history_base['test_acc'], marker='s', color='#ef4444', linestyle='--', linewidth=2.5, markersize=8, label='HAL + Mean Pooling (Baseline)')
    plt.title('Figure 1: Test Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Training Loss Convergence
    plt.figure(figsize=(8, 5.5))
    plt.plot(range(1, len(history_attn['train_loss']) + 1), history_attn['train_loss'], marker='o', color='#10b981', linewidth=2.5, markersize=8, label='HAL + Attention (Proposed)')
    plt.plot(range(1, len(history_base['train_loss']) + 1), history_base['train_loss'], marker='s', color='#ef4444', linestyle='--', linewidth=2.5, markersize=8, label='HAL + Mean Pooling (Baseline)')
    plt.title('Figure 2: Training Loss Convergence', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.savefig('loss_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Training Accuracy Convergence
    plt.figure(figsize=(8, 5.5))
    plt.plot(range(1, len(history_attn['train_acc']) + 1), history_attn['train_acc'], marker='o', color='#3b82f6', linewidth=2.5, markersize=8, label='HAL + Attention (Proposed)')
    plt.plot(range(1, len(history_base['train_acc']) + 1), history_base['train_acc'], marker='s', color='#f97316', linestyle='--', linewidth=2.5, markersize=8, label='HAL + Mean Pooling (Baseline)')
    plt.title('Figure 3: Training Accuracy Convergence', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Training Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Train Accuracy (%)', fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.savefig('train_accuracy_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('[INFO] Comparative plots saved successfully as PNG files.')

def main():
    set_seed(42)
    
    MAX_VOCAB = 10000
    MAX_LEN = 200
    BATCH_SIZE = 64
    EMBED_DIM = 300
    EPOCHS = 30
    PATIENCE = 5
    LR = 0.0005
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Utilizing device: {device}')
    
    train_data, test_data, word2idx = load_and_prepare_data(max_vocab=MAX_VOCAB)
    hal_tensor = build_hal_matrix(train_data, word2idx, window_size=5, embed_dim=EMBED_DIM)
    
    print('\n[INFO] Initializing DataLoaders...')
    train_loader = create_dataloaders(train_data, word2idx, max_len=MAX_LEN, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = create_dataloaders(test_data, word2idx, max_len=MAX_LEN, batch_size=128, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    base_model = BaselineHALMeanPooling(hal_tensor, embed_dim=EMBED_DIM).to(device)
    base_opt = optim.Adam(base_model.parameters(), lr=LR, weight_decay=1e-4)
    base_model, history_base = train_and_evaluate(
        base_model, base_opt, criterion, train_loader, test_loader, word2idx, device, epochs=EPOCHS, patience=PATIENCE, model_name='Baseline (HAL + Mean Pooling)'
    )
    
    attn_model = RobustHALAttention(hal_tensor, embed_dim=EMBED_DIM).to(device)
    attn_opt = optim.Adam(attn_model.parameters(), lr=LR, weight_decay=1e-4)
    attn_model, history_attn = train_and_evaluate(
        attn_model, attn_opt, criterion, train_loader, test_loader, word2idx, device, epochs=EPOCHS, patience=PATIENCE, model_name='Proposed (HAL + Attention)'
    )
    
    plot_results(history_base, history_attn)
    
    # 💾 The returned 'attn_model' is guaranteed to be the best performing model
    torch.save(attn_model.state_dict(), 'hal_attention_best.pth')
    print('\n[INFO] Best model weights successfully saved to hal_attention_best.pth')
    print('[INFO] End-to-end training pipeline concluded.')

if __name__ == '__main__':
    main()