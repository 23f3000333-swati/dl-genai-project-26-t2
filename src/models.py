import torch
import torch.nn as nn
from transformers import AutoModelForMultipleChoice
from peft import LoraConfig, get_peft_model, TaskType

class RegularizedMCQClassifier(nn.Module):
    """Model 1: Custom PyTorch Neural Network with Regularization to prevent overfitting."""
    def __init__(self, input_dim: int, num_classes: int = 5):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def get_deberta_lora_model(model_name: str = "microsoft/deberta-v3-base"):
    """Model 2: Pretrained DeBERTa wrapped with Parameter-Efficient LoRA Adapters."""
    base_model = AutoModelForMultipleChoice.from_pretrained(model_name)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query_proj", "value_proj"]
    )
    
    model = get_peft_model(base_model, peft_config)
    return model