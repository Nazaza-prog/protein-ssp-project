import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

# Імпорти з твого проекту
from src.data.dataset import create_dataloaders
from src.models.cnn_lstm_model import create_model
from src.evaluation.metrics import MetricsTracker

# Налаштування matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def evaluate_model(model, test_loader, device):
    """
    Оцінює модель на test set і повертає детальні метрики
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0
    
    print("\n🔍 Оцінка моделі на тестовому наборі...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs.view(-1, 3), labels.view(-1))
            total_loss += loss.item()
            
            # Збираємо передбачення
            predictions = torch.argmax(outputs, dim=-1)
            
            # Фільтруємо padding (-100)
            mask = labels != -100
            
            all_predictions.extend(predictions[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    return all_predictions, all_labels, avg_loss


def calculate_metrics(predictions, labels):
    """
    Розраховує детальні метрики
    """
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Класи
    class_names = ['Helix (H)', 'Sheet (E)', 'Coil (C)']
    
    # Загальна точність
    accuracy = (predictions == labels).mean() * 100
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Детальний звіт
    report = classification_report(labels, predictions, 
                                   target_names=class_names,
                                   output_dict=True)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'report': report,
        'class_names': class_names
    }


def create_confusion_matrix_plot(cm, class_names, save_path):
    """
    Створює красивий confusion matrix
    """
    # Нормалізуємо
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Тепловий графік
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Відсоток (%)'},
                linewidths=1,
                linecolor='gray',
                ax=ax)
    
    # Додаємо кількості у відсотках
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            percentage = cm_normalized[i, j]
            ax.text(j + 0.5, i + 0.7, f'({count})',
                   ha='center', va='center', fontsize=9, color='gray')
    
    ax.set_ylabel('Справжній клас', fontsize=14, fontweight='bold')
    ax.set_xlabel('Передбачений клас', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix (Матриця помилок)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {save_path.name}")


def create_accuracy_plot(metrics, save_path):
    """
    Графік точності по класах
    """
    class_names = metrics['class_names']
    report = metrics['report']
    
    # Витягуємо accuracy по класах (f1-score як проксі)
    accuracies = []
    supports = []
    
    for i, class_name in enumerate(class_names):
        key = class_name
        accuracies.append(report[key]['f1-score'] * 100)
        supports.append(report[key]['support'])
    
    # Додаємо середнє
    class_names_extended = class_names + ['Середнє (Q3)']
    accuracies.append(metrics['accuracy'])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax.bar(class_names_extended, accuracies, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Значення над стовпчиками
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Додаємо підтримку під назвами класів
    new_labels = []
    for i, name in enumerate(class_names_extended):
        if i < len(supports):
            new_labels.append(f'{name}\n(n={supports[i]})')
        else:
            new_labels.append(name)
    
    ax.set_xticklabels(new_labels, fontsize=11)
    ax.set_ylabel('Точність (%)', fontsize=14, fontweight='bold')
    ax.set_title('Точність моделі прогнозування вторинної структури білків', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Горизонтальна лінія
    ax.axhline(y=metrics['accuracy'], color='red', linestyle='--', 
               linewidth=2, alpha=0.6, label=f'Q3 = {metrics["accuracy"]:.2f}%')
    ax.legend(fontsize=12, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {save_path.name}")


def create_metrics_comparison(metrics, save_path):
    """
    Порівняння Precision, Recall, F1-Score
    """
    class_names = metrics['class_names']
    report = metrics['report']
    
    precision_vals = []
    recall_vals = []
    f1_vals = []
    
    for class_name in class_names:
        precision_vals.append(report[class_name]['precision'] * 100)
        recall_vals.append(report[class_name]['recall'] * 100)
        f1_vals.append(report[class_name]['f1-score'] * 100)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, precision_vals, width, label='Precision', 
                   color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, recall_vals, width, label='Recall', 
                   color='#4ECDC4', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, f1_vals, width, label='F1-Score', 
                   color='#45B7D1', alpha=0.8, edgecolor='black')
    
    # Значення над стовпчиками
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_ylabel('Значення (%)', fontsize=14, fontweight='bold')
    ax.set_title('Порівняння метрик класифікації по класах', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {save_path.name}")


def create_summary_report(metrics, save_path):
    """
    Комбінований звіт
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    class_names = metrics['class_names']
    report = metrics['report']
    cm = metrics['confusion_matrix']
    
    # 1. Confusion Matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=[c.split()[0] for c in class_names],
                yticklabels=[c.split()[0] for c in class_names],
                cbar_kws={'label': '%'}, ax=ax1)
    ax1.set_title('A) Confusion Matrix', fontsize=12, fontweight='bold', loc='left')
    ax1.set_ylabel('Справжній клас', fontsize=10)
    ax1.set_xlabel('Передбачений клас', fontsize=10)
    
    # 2. Точність по класах
    ax2 = fig.add_subplot(gs[0, 1])
    accuracies = [report[cn]['f1-score'] * 100 for cn in class_names]
    accuracies.append(metrics['accuracy'])
    labels = [c.split()[0] for c in class_names] + ['Q3']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    bars = ax2.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black')
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax2.set_ylabel('Точність (%)', fontsize=10)
    ax2.set_title('B) Точність моделі', fontsize=12, fontweight='bold', loc='left')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Метрики порівняння
    ax3 = fig.add_subplot(gs[1, :])
    precision = [report[cn]['precision'] * 100 for cn in class_names]
    recall = [report[cn]['recall'] * 100 for cn in class_names]
    f1 = [report[cn]['f1-score'] * 100 for cn in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    ax3.bar(x - width, precision, width, label='Precision', color='#FF6B6B', alpha=0.8)
    ax3.bar(x, recall, width, label='Recall', color='#4ECDC4', alpha=0.8)
    ax3.bar(x + width, f1, width, label='F1-Score', color='#45B7D1', alpha=0.8)
    
    ax3.set_ylabel('Значення (%)', fontsize=10)
    ax3.set_title('C) Порівняння метрик класифікації', fontsize=12, fontweight='bold', loc='left')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, fontsize=10)
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 105)
    ax3.grid(axis='y', alpha=0.3)
    
    fig.suptitle(f'Звіт про результати моделі (Q3 = {metrics["accuracy"]:.2f}%)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {save_path.name}")


def save_text_report(metrics, save_path):
    """
    Зберігає текстовий звіт
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ЗВІТ ПРО РЕЗУЛЬТАТИ МОДЕЛІ\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"📊 ЗАГАЛЬНА ТОЧНІСТЬ: Q3 = {metrics['accuracy']:.2f}%\n\n")
        
        f.write("-"*70 + "\n")
        f.write("ДЕТАЛЬНІ МЕТРИКИ ПО КЛАСАХ:\n")
        f.write("-"*70 + "\n\n")
        
        report = metrics['report']
        
        for class_name in metrics['class_names']:
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {report[class_name]['precision']*100:.2f}%\n")
            f.write(f"  Recall:    {report[class_name]['recall']*100:.2f}%\n")
            f.write(f"  F1-Score:  {report[class_name]['f1-score']*100:.2f}%\n")
            f.write(f"  Support:   {report[class_name]['support']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("CONFUSION MATRIX:\n")
        f.write("-"*70 + "\n\n")
        
        cm = metrics['confusion_matrix']
        f.write("             Pred_H  Pred_E  Pred_C\n")
        for i, class_name in enumerate(['True_H', 'True_E', 'True_C']):
            f.write(f"{class_name:8s}  ")
            for j in range(3):
                f.write(f"{cm[i,j]:7d} ")
            f.write("\n")
    
    print(f"  ✓ {save_path.name}")


def main():
    """
    Головна функція
    """
    print("\n" + "="*70)
    print("ВІЗУАЛІЗАЦІЯ РЕАЛЬНИХ РЕЗУЛЬТАТІВ МОДЕЛІ")
    print("="*70 + "\n")
    
    # Завантаження конфігурації
    with open('configs/baseline_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    # Створення датасетів
    print("📚 Завантаження даних...")
    _, test_loader = create_dataloaders(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_cb513_path'],
        batch_size=config['training']['batch_size'],
        max_len=config['data']['max_seq_length'],
        use_q3=True,
        num_workers=0
    )
    
    # Завантаження моделі
    print("\n🤖 Завантаження моделі...")
    model = create_model(
        model_type=config['model']['type'],
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )
    
    # Завантаження best model
    best_model_path = Path('results/baseline/best_model.pt')
    if not best_model_path.exists():
        print(f"❌ Файл {best_model_path} не знайдено!")
        print("Переконайся що тренування завершено і файл існує.")
        return
    
    checkpoint = torch.load(best_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✓ Модель завантажено (epoch {checkpoint['epoch']})")
    
    # Оцінка моделі
    predictions, labels, test_loss = evaluate_model(model, test_loader, device)
    
    print(f"\n✓ Оцінка завершена!")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Всього передбачень: {len(predictions):,}")
    
    # Розрахунок метрик
    print("\n📊 Розрахунок метрик...")
    metrics = calculate_metrics(predictions, labels)
    
    print(f"\n🎯 РЕЗУЛЬТАТИ:")
    print(f"  Q3 Accuracy: {metrics['accuracy']:.2f}%")
    
    # Створення директорії для візуалізацій
    viz_dir = Path('results/baseline/visualizations')
    viz_dir.mkdir(exist_ok=True)
    
    print(f"\n🎨 Створення візуалізацій...\n")
    
    # Створення графіків
    create_confusion_matrix_plot(
        metrics['confusion_matrix'], 
        metrics['class_names'],
        viz_dir / 'confusion_matrix.png'
    )
    
    create_accuracy_plot(
        metrics,
        viz_dir / 'accuracy_plot.png'
    )
    
    create_metrics_comparison(
        metrics,
        viz_dir / 'metrics_comparison.png'
    )
    
    create_summary_report(
        metrics,
        viz_dir / 'summary_report.png'
    )
    
    save_text_report(
        metrics,
        viz_dir / 'results_report.txt'
    )
    
    print("\n" + "="*70)
    print("✅ ВІЗУАЛІЗАЦІЯ ЗАВЕРШЕНА!")
    print("="*70 + "\n")
    
    print(f"📁 Файли збережені в: {viz_dir}\n")
    print("Створені файли:")
    print("  1. confusion_matrix.png - Матриця помилок")
    print("  2. accuracy_plot.png - Графік точності")
    print("  3. metrics_comparison.png - Порівняння метрик")
    print("  4. summary_report.png - Комбінований звіт (ВСЕ В ОДНОМУ)")
    print("  5. results_report.txt - Текстовий звіт")
    
    print("\n💡 Рекомендації:")
    print("  - Для презентації: summary_report.png")
    print("  - Для звіту: всі окремі графіки")
    print("  - Для аналізу помилок: confusion_matrix.png")


if __name__ == "__main__":
    main()
