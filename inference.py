import torch
import re
from models import RobustHALAttention
from data_utils import load_and_prepare_data, build_hal_matrix

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.split()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Initializing Inference Pipeline...")
    
    # 1. إعادة بناء القاموس ومصفوفة HAL (بسرعة لاستخدامها في التضمين)
    train_data, _, word2idx = load_and_prepare_data(max_vocab=10000)
    hal_tensor = build_hal_matrix(train_data, word2idx, window_size=5, embed_dim=300)
    
    # 2. تحميل النموذج الفائز
    model = RobustHALAttention(hal_tensor, embed_dim=300).to(device)
    try:
        model.load_state_dict(torch.load("hal_attention_best.pth", map_location=device))
        print("[INFO] Successfully loaded 'hal_attention_best.pth'")
    except FileNotFoundError:
        print("[ERROR] Model weights not found. Please run train.py first.")
        return

    model.eval()
    
    # 3. الجملة المراد اختبارها
    sample_text = "the cinematography was brilliant but the acting was completely awful and ruined the experience"
    print(f"\n[TEXT] {sample_text}")
    
    words = clean_text(sample_text)
    indices =[word2idx.get(w, word2idx["<UNK>"]) for w in words]
    
    x_tensor = torch.LongTensor([indices]).to(device)
    mask = (x_tensor == word2idx["<PAD>"])
    
    with torch.no_grad():
        logits, alpha = model(x_tensor, mask)
        prediction = torch.argmax(logits, dim=1).item()
        
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"[PREDICTION] {sentiment}")
    print("\n[ATTENTION WEIGHTS]")
    
    weights = alpha[0].squeeze().cpu().numpy()
    word_weight_pairs = list(zip(words, weights))
    word_weight_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for word, weight in word_weight_pairs:
        if weight > 0.01: # إظهار الكلمات المؤثرة فقط
            print(f"{word:>15} : {weight:.4f}")

if __name__ == "__main__":
    main()
