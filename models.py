import torch
import torch.nn as nn

class BaselineHALMeanPooling(nn.Module):
    """
    النموذج الكلاسيكي (1996) الذي يستخدم المتوسط الحسابي (Mean Pooling)
    Baseline model representing traditional HAL aggregation.
    """
    def __init__(self, embeddings, embed_dim=300, num_classes=2):
        super(BaselineHALMeanPooling, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, mask):
        embedded = self.embedding(x)
        # إخفاء الـ Padding
        expanded_mask = mask.unsqueeze(-1).expand(embedded.size())
        embedded_masked = embedded.masked_fill(expanded_mask, 0.0)
        # حساب المتوسط (Mean Pooling)
        lengths = torch.clamp((~mask).sum(dim=1, keepdim=True).float(), min=1.0)
        sentence_vec = torch.sum(embedded_masked, dim=1) / lengths
        return self.classifier(sentence_vec)


class RobustHALAttention(nn.Module):
    """
    نموذجك المبتكر الذي يدمج الانتباه مع مصفوفة HAL
    Proposed model: HAL + Temperature-Scaled Attention Pooling
    """
    def __init__(self, embeddings, embed_dim=300, num_classes=2):
        super(RobustHALAttention, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )
        
        self.temp = 2.0 # Temperature Scaling لمنع الـ Overfitting

    def forward(self, x, mask):
        embedded = self.embedding(x)
        
        # حساب أوزان الانتباه
        attn_scores = self.attention(embedded)
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)
        alpha = torch.softmax(attn_scores / self.temp, dim=1)
        
        # التجميع بالانتباه (Attention Pooling)
        sentence_vec = torch.sum(alpha * embedded, dim=1)
        logits = self.classifier(sentence_vec)
        
        return logits, alpha
