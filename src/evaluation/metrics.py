"""
Метрики для оцінки моделей Secondary Structure Prediction
Оновлено для NetSurfP-3.0 датасету
"""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from typing import Dict

def q3_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Обчислює Q3 accuracy, ігноруючи padding (-100)
    
    Параметри:
        predictions: [batch, seq_len, num_classes] або [batch, seq_len]
        targets: [batch, seq_len]
    
    Повертає:
        accuracy: 0-100%
    """
    # Якщо predictions має 3 виміри - беремо argmax
    if predictions.dim() == 3:
        preds = predictions.argmax(dim=-1)
    else:
        preds = predictions
    
    # Маска валідних позицій (не padding)
    valid = targets != -100
    
    if valid.sum() == 0:
        return 0.0
    
    # Підрахунок правильних
    correct = ((preds == targets) & valid).sum().float()
    total = valid.sum().float()
    
    return (correct / total * 100).item()


def per_class_accuracy(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    num_classes: int = 3
) -> Dict:
    """
    Accuracy для кожного класу окремо
    
    Повертає:
        dict з accuracy для H, E, C (або 8 класів для Q8)
    """
    if predictions.dim() == 3:
        preds = predictions.argmax(dim=-1)
    else:
        preds = predictions
    
    valid = targets != -100
    
    results = {}
    class_names = ['H (Helix)', 'E (Sheet)', 'C (Coil)'] if num_classes == 3 else \
                  ['G', 'H', 'I', 'E', 'B', 'T', 'S', 'C']
    
    for c in range(num_classes):
        class_mask = (targets == c) & valid
        
        if class_mask.sum() > 0:
            correct = ((preds == c) & class_mask).sum().float()
            acc = (correct / class_mask.sum() * 100).item()
            results[class_names[c]] = {
                'accuracy': acc,
                'support': class_mask.sum().item()
            }
        else:
            results[class_names[c]] = {
                'accuracy': 0.0,
                'support': 0
            }
    
    return results


def confusion_matrix_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3
) -> np.ndarray:
    """
    Обчислює confusion matrix
    
    Повертає:
        confusion matrix [num_classes, num_classes]
    """
    if predictions.dim() == 3:
        preds = predictions.argmax(dim=-1)
    else:
        preds = predictions
    
    valid = targets != -100
    
    # Flatten та фільтруємо padding
    preds_flat = preds[valid].cpu().numpy()
    targets_flat = targets[valid].cpu().numpy()
    
    return confusion_matrix(targets_flat, preds_flat, labels=range(num_classes))


class MetricsTracker:
    """
    Клас для накопичення та обчислення метрик протягом епохи
    """
    
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.class_names = ['H', 'E', 'C'] if num_classes == 3 else list('GHIBESTC')
        self.reset()
    
    def reset(self):
        """Скидає накопичені значення"""
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Додає нові прогнози
        
        Параметри:
            predictions: [batch, seq_len, num_classes] або [batch, seq_len]
            targets: [batch, seq_len]
        """
        if predictions.dim() == 3:
            preds = predictions.argmax(dim=-1)
        else:
            preds = predictions
        
        self.all_predictions.append(preds.cpu())
        self.all_targets.append(targets.cpu())
    
    def compute(self) -> Dict:
        """
        Обчислює всі метрики
        
        Повертає:
            dict з метриками
        """
        # Об'єднуємо всі батчі
        predictions = torch.cat(self.all_predictions, dim=0)
        targets = torch.cat(self.all_targets, dim=0)
        
        # Q3/Q8 accuracy
        accuracy = q3_accuracy(predictions, targets)
        
        # Per-class метрики
        class_metrics = per_class_accuracy(predictions, targets, self.num_classes)
        
        # Confusion matrix
        conf_matrix = confusion_matrix_metrics(predictions, targets, self.num_classes)
        
        return {
            'accuracy': accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': conf_matrix
        }
    
    def print_results(self, results: Dict = None):
        """Друкує результати в зручному форматі"""
        if results is None:
            results = self.compute()
        
        mode = f"Q{self.num_classes}" if self.num_classes in [3, 8] else "Accuracy"
        
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТИ ОЦІНКИ ({mode})")
        print(f"{'='*60}")
        
        # Загальна точність
        print(f"\n{mode} Accuracy: {results['accuracy']:.2f}%")
        
        # Per-class метрики
        print(f"\nТочність по класах:")
        print(f"{'─'*60}")
        print(f"{'Клас':<15} {'Accuracy':<12} {'Підтримка':<12}")
        print(f"{'─'*60}")
        
        for cls_name, metrics in results['class_metrics'].items():
            print(f"{cls_name:<15} {metrics['accuracy']:>6.2f}%      "
                  f"{metrics['support']:>8}")
        
        print(f"{'─'*60}")
        
        # Середня accuracy (macro)
        avg_acc = np.mean([m['accuracy'] for m in results['class_metrics'].values()])
        print(f"{'Macro Avg':<15} {avg_acc:>6.2f}%")
        
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Тестування метрик
    print("=== Тестування Метрик ===\n")
    
    # Генеруємо тестові дані
    np.random.seed(42)
    n_samples = 1000
    true_labels = np.random.randint(0, 3, n_samples)
    predictions = true_labels.copy()
    
    # Додаємо трохи помилок
    error_indices = np.random.choice(n_samples, size=100, replace=False)
    predictions[error_indices] = np.random.randint(0, 3, 100)
    
    # Обчислюємо метрики
    calculator = MetricsCalculator(num_classes=3)
    calculator.update(predictions, true_labels)
    results = calculator.compute()
    calculator.print_results(results)