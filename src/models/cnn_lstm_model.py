"""
CNN + BiGRU модель для прогнозування вторинної структури білків
Оптимізована для CPU тренування
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSPModel(nn.Module):
    """
    Hybrid CNN + BiGRU архітектура для SSP
    
    Архітектура:
        1. CNN частина - витягує локальні паттерни (мотиви структури)
        2. BiGRU частина - захоплює довгострокові залежності
        3. Fully connected - фінальна класифікація
    
    Параметри:
        input_dim: кількість вхідних features (50 для AA+HMM)
        hidden_dim: розмір прихованих шарів (128 оптимально для CPU)
        num_classes: кількість класів (3 для Q3, 8 для Q8)
        dropout: коефіцієнт dropout для регуляризації
    """
    
    def __init__(
        self, 
        input_dim: int = 50,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super(SSPModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # CNN частина - витягує локальні features
        # Conv1D працює швидше за Conv2D на CPU
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        # Batch Normalization для стабільності
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Bidirectional GRU (швидше за LSTM на CPU)
        # 2 шари для балансу між якістю та швидкістю
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Dropout для регуляризації
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected для фінальної класифікації
        # *2 бо BiGRU (forward + backward)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Параметри:
            x: [batch_size, seq_len, input_dim]
        
        Повертає:
            [batch_size, seq_len, num_classes]
        """
        # Транспонуємо для Conv1D: [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # CNN блок 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # CNN блок 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # CNN блок 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Повертаємо назад: [batch, seq_len, channels]
        x = x.transpose(1, 2)
        
        # BiGRU
        x, _ = self.gru(x)
        x = self.dropout(x)
        
        # Фінальна класифікація
        x = self.fc(x)
        
        return x
    
    def count_parameters(self):
        """Підраховує кількість тренувальних параметрів"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleCNN(nn.Module):
    """
    Спрощена CNN модель для швидкого базового експерименту
    Швидша за SSPModel але трохи менш точна
    """
    
    def __init__(
        self,
        input_dim: int = 50,
        num_classes: int = 3,
        dropout: float = 0.3
    ):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        x = self.fc(x)
        
        return x


def create_model(
    model_type: str = 'hybrid',
    input_dim: int = 50,
    hidden_dim: int = 128,
    num_classes: int = 3,
    dropout: float = 0.3
):
    """
    Фабрика для створення моделей
    
    Параметри:
        model_type: 'hybrid' (CNN+GRU) або 'simple' (тільки CNN)
        input_dim: 50 для AA+HMM
        hidden_dim: 128 оптимально для CPU
        num_classes: 3 для Q3, 8 для Q8
        dropout: 0.3-0.4 добре працює
    
    Повертає:
        model, інформація про параметри
    """
    if model_type == 'hybrid':
        model = SSPModel(input_dim, hidden_dim, num_classes, dropout)
    elif model_type == 'simple':
        model = SimpleCNN(input_dim, num_classes, dropout)
    else:
        raise ValueError(f"Невідомий тип моделі: {model_type}")
    
    num_params = model.count_parameters() if hasattr(model, 'count_parameters') else \
                 sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Створено модель: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Параметри:")
    print(f"  Вхідна розмірність: {input_dim}")
    print(f"  Прихована розмірність: {hidden_dim}")
    print(f"  Кількість класів: {num_classes}")
    print(f"  Dropout: {dropout}")
    print(f"\nКількість параметрів: {num_params:,}")
    print(f"Розмір моделі: ~{num_params * 4 / (1024**2):.2f} MB")
    print(f"{'='*60}\n")
    
    return model


if __name__ == "__main__":
    # Тестування моделей
    print("\n=== ТЕСТУВАННЯ МОДЕЛЕЙ ===\n")
    
    # Тестові дані
    batch_size = 4
    seq_len = 100
    input_dim = 50
    num_classes = 3
    
    x = torch.randn(batch_size, seq_len, input_dim)
    print(f"Вхідна форма: {x.shape}")
    
    # Тест Hybrid моделі
    print("\n1. Hybrid CNN+BiGRU модель:")
    model_hybrid = create_model('hybrid', input_dim, 128, num_classes, 0.3)
    
    output = model_hybrid(x)
    print(f"Вихідна форма: {output.shape}")
    print(f"Очікувана форма: [{batch_size}, {seq_len}, {num_classes}]")
    
    # Тест Simple CNN моделі
    print("\n2. Simple CNN модель:")
    model_simple = create_model('simple', input_dim, num_classes=num_classes)
    
    output = model_simple(x)
    print(f"Вихідна форма: {output.shape}")
    
    print("\n✓ Всі моделі працюють коректно!")
