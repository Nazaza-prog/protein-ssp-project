"""
Тренування моделі SSP на NetSurfP-3.0
Фінальна версія з усіма оптимізаціями + автозбереження кожної епохи
Автор: МАН проект з інформатики

Розташування: src/training/train.py
"""
import torch
import torch.nn as nn
import yaml
import signal
import sys
from pathlib import Path
from tqdm import tqdm

# Знаходимо кореневу директорію проєкту (для роботи з src/training/)
root_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root_dir))

# Наші модулі
from src.data.dataset import create_dataloaders
from src.models.cnn_lstm_model import create_model
from src.evaluation.metrics import MetricsTracker
from src.utils.config import (
    setup_cpu_optimization, 
    set_seed, 
    get_device,
    safe_save_checkpoint
)

# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================
stop_training = False

def signal_handler(sig, frame):
    """Обробник Ctrl+C для graceful shutdown"""
    global stop_training
    
    if stop_training:
        print("\n\n🛑 ПРИМУСОВА ЗУПИНКА!")
        print("Модель НЕ збережено!")
        sys.exit(0)
    
    print("\n\n⚠️  Отримано сигнал зупинки (Ctrl+C)")
    print("Завершення поточної епохи та збереження моделі...")
    print("(Натисніть Ctrl+C ще раз для примусової зупинки)")
    stop_training = True

signal.signal(signal.SIGINT, signal_handler)

# ============================================================================
# CPU ОПТИМІЗАЦІЯ
# ============================================================================
setup_cpu_optimization(num_threads=12)

# ============================================================================
# ФУНКЦІЇ ТРЕНУВАННЯ
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Один epoch тренування"""
    model.train()
    total_loss = 0
    metrics = MetricsTracker(num_classes=3)
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs.view(-1, 3), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        metrics.update(outputs.detach(), labels)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    results = metrics.compute()
    avg_loss = total_loss / len(train_loader)
    return avg_loss, results['accuracy']

def validate_epoch(model, val_loader, criterion, device):
    """Валідація"""
    model.eval()
    total_loss = 0
    metrics = MetricsTracker(num_classes=3)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            outputs = model(sequences)
            loss = criterion(outputs.view(-1, 3), labels.view(-1))
            total_loss += loss.item()
            metrics.update(outputs, labels)
    
    results = metrics.compute()
    avg_loss = total_loss / len(val_loader)
    return avg_loss, results

def save_checkpoint_safely(epoch, model, optimizer, best_val_acc, config, save_dir, checkpoint_type='auto'):
    """
    Єдина функція для збереження всіх типів checkpoint
    
    Параметри:
        epoch: номер епохи
        model: модель
        optimizer: optimizer
        best_val_acc: найкраща точність
        config: конфігурація
        save_dir: директорія для збереження
        checkpoint_type: тип checkpoint ('auto', 'best', 'archive', 'interrupted')
    """
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'config': config,
        'checkpoint_type': checkpoint_type
    }
    
    if checkpoint_type == 'auto':
        # Автозбереження після кожної епохи - з backup попередньої
        file_path = save_dir / 'last_epoch.pt'
        success = safe_save_checkpoint(checkpoint_data, file_path, backup=True)
        if success:
            print(f"  💾 Автозбереження: last_epoch.pt (epoch {epoch})")
            if (save_dir / 'last_epoch_backup.pt').exists():
                print(f"     └─ Backup: last_epoch_backup.pt (epoch {epoch-1})")
    
    elif checkpoint_type == 'best':
        # Найкраща модель - БЕЗ backup (сама є найкращою)
        file_path = save_dir / 'best_model.pt'
        checkpoint_data['val_acc'] = checkpoint_data['best_val_acc']
        success = safe_save_checkpoint(checkpoint_data, file_path, backup=False)
        if success:
            print(f"  ✓ Збережено найкращу модель (Q3: {best_val_acc:.2f}%)")
    
    elif checkpoint_type == 'archive':
        # Архівне збереження - без backup
        file_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
        success = safe_save_checkpoint(checkpoint_data, file_path, backup=False)
        if success:
            print(f"  📦 Архівне збереження: checkpoint_epoch_{epoch}.pt")
    
    elif checkpoint_type == 'interrupted':
        # При Ctrl+C або Early Stopping
        checkpoint_data['stopped_manually'] = True
        file_path = save_dir / 'interrupted_checkpoint.pt'
        success = safe_save_checkpoint(checkpoint_data, file_path, backup=False)
        if success:
            print(f"  ✓ Checkpoint збережено: interrupted_checkpoint.pt")
    
    return success

# ============================================================================
# ГОЛОВНА ФУНКЦІЯ
# ============================================================================
def main():
    print("\n" + "="*70)
    print("ТРЕНУВАННЯ МОДЕЛІ ПРОГНОЗУВАННЯ ВТОРИННОЇ СТРУКТУРИ БІЛКІВ")
    print("="*70 + "\n")
    
    # Завантаження конфігурації
    with open(root_dir / 'configs' / 'baseline_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Налаштування
    set_seed(config['seed'])
    device = get_device(prefer_cuda=False)
    print()
    
    # DataLoaders
    print("Завантаження даних...")
    train_loader, test_loader = create_dataloaders(
        train_path=str(root_dir / config['data']['train_path']),
        test_path=str(root_dir / config['data']['test_cb513_path']),
        batch_size=config['training']['batch_size'],
        max_len=config['data']['max_seq_length'],
        use_q3=True,
        num_workers=config['cpu']['num_workers']
    )
    
    # Модель
    print("\nСтворення моделі...")
    model = create_model(
        model_type=config['model']['type'],
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)
    
    # Компіляція (якщо доступна)
    if sys.version_info < (3, 14) and hasattr(torch, 'compile'):
        try:
            print("Компіляція моделі...")
            model = torch.compile(model, mode='reduce-overhead')
            print("✓ Модель скомпільована")
        except Exception as e:
            print(f"⚠️  Компіляція не вдалася: {e}")
    else:
        print("ℹ️  torch.compile() недоступний (Python 3.14+)")
    
    # Loss та Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    if config['training']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    
    # Scheduler
    scheduler = None
    if config['training']['scheduler']['use']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=config['training']['scheduler']['patience'],
            factor=config['training']['scheduler']['factor'],
            min_lr=config['training']['scheduler']['min_lr']
        )
        print("✓ Learning rate scheduler створено")
    
    # Папка для збереження
    save_dir = root_dir / config['logging']['log_dir']
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # ЦИКЛ ТРЕНУВАННЯ
    # ========================================================================
    print("\n" + "="*70)
    print("ПОЧАТОК ТРЕНУВАННЯ")
    print("="*70)
    print(f"\n📁 Директорія для збереження: {save_dir}")
    print(f"💾 Автозбереження: Після кожної епохи → last_epoch.pt")
    print(f"🔄 Backup системи: last_epoch_backup.pt (попередня епоха)")
    print(f"⭐ Найкраща модель: best_model.pt")
    print(f"📦 Архівні checkpoint: кожні 5 епох\n")
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Перевірка graceful shutdown
        global stop_training
        if stop_training:
            print("\n" + "="*70)
            print("ТРЕНУВАННЯ ЗУПИНЕНО КОРИСТУВАЧЕМ")
            print("="*70)
            
            save_checkpoint_safely(
                epoch - 1, model, optimizer, best_val_acc, 
                config, save_dir, checkpoint_type='interrupted'
            )
            
            print(f"\nОстанній завершений epoch: {epoch - 1}")
            print(f"Найкраща валідаційна точність: {best_val_acc:.2f}%")
            print("\nДля продовження: python resume_training.py")
            break
        
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print("-" * 70)
        
        # Тренування
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Валідація
        val_loss, val_results = validate_epoch(
            model, test_loader, criterion, device
        )
        val_acc = val_results['accuracy']
        
        # Друк результатів
        print(f"\nРезультати епохи {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Q3: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Q3:   {val_acc:.2f}%")
        
        print(f"\n  Точність по класах (Validation):")
        for cls_name, metrics in val_results['class_metrics'].items():
            print(f"    {cls_name}: {metrics['accuracy']:.2f}%")
        
        # Scheduler
        if scheduler:
            scheduler.step(val_acc)
        
        # ====================================================================
        # ЗБЕРЕЖЕННЯ МОДЕЛЕЙ
        # ====================================================================
        print(f"\n  💾 Збереження checkpoint...")
        
        # 1. АВТОЗБЕРЕЖЕННЯ ПІСЛЯ КОЖНОЇ ЕПОХИ (ОБОВ'ЯЗКОВО!)
        save_checkpoint_safely(
            epoch, model, optimizer, best_val_acc,
            config, save_dir, checkpoint_type='auto'
        )
        
        # 2. Найкраща модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_checkpoint_safely(
                epoch, model, optimizer, best_val_acc,
                config, save_dir, checkpoint_type='best'
            )
        else:
            patience_counter += 1
        
        # 3. Архівне збереження кожні 5 епох
        if epoch % 5 == 0:
            save_checkpoint_safely(
                epoch, model, optimizer, best_val_acc,
                config, save_dir, checkpoint_type='archive'
            )
        
        # ====================================================================
        # EARLY STOPPING
        # ====================================================================
        if config['training']['early_stopping']['use']:
            if patience_counter >= config['training']['early_stopping']['patience']:
                print(f"\n⚠️  Early stopping після {epoch} епох")
                print(f"Найкраща валідаційна точність: {best_val_acc:.2f}%")
                
                # Завантажуємо найкращу модель
                best_checkpoint = torch.load(save_dir / 'best_model.pt')
                
                # Зберігаємо final checkpoint з найкращою моделлю
                final_checkpoint_data = {
                    'epoch': best_checkpoint['epoch'],
                    'model_state_dict': best_checkpoint['model_state_dict'],
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'config': config,
                    'early_stopped': True,
                    'stopped_at_epoch': epoch
                }
                
                safe_save_checkpoint(
                    final_checkpoint_data,
                    save_dir / 'final_checkpoint.pt',
                    backup=False
                )
                
                print(f"✓ Final checkpoint (найкраща модель з epoch {best_checkpoint['epoch']})")
                break
    
    # ========================================================================
    # ФІНАЛЬНА ОЦІНКА
    # ========================================================================
    print("\n" + "="*70)
    print("ФІНАЛЬНА ОЦІНКА НА ТЕСТОВОМУ НАБОРІ")
    print("="*70 + "\n")
    
    best_model_path = save_dir / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        _, final_results = validate_epoch(model, test_loader, criterion, device)
        
        print("\nФінальні результати:")
        print(f"  Q3 Accuracy: {final_results['accuracy']:.2f}%")
        print(f"\n  Точність по класах:")
        for cls_name, metrics in final_results['class_metrics'].items():
            print(f"    {cls_name}: {metrics['accuracy']:.2f}% "
                  f"(підтримка: {metrics['support']})")
        
        print("\n" + "="*70)
        print("✓ ТРЕНУВАННЯ ЗАВЕРШЕНО УСПІШНО!")
        print("="*70 + "\n")
        
        print(f"📁 Збережені файли:")
        print(f"  ⭐ Найкраща модель: {best_model_path}")
        print(f"  💾 Остання епоха:   {save_dir / 'last_epoch.pt'}")
        if (save_dir / 'last_epoch_backup.pt').exists():
            print(f"  🔄 Backup епохи:    {save_dir / 'last_epoch_backup.pt'}")
        
        print(f"\n🎯 Найкраща точність: Q3 = {best_val_acc:.2f}%")
        
        # Причина завершення
        if stop_training:
            print("\nПричина: Зупинено користувачем (Ctrl+C)")
        elif patience_counter >= config['training']['early_stopping']['patience']:
            print("\nПричина: Early Stopping")
        else:
            print("\nПричина: Досягнуто максимум епох")
        
        print("\n💡 Для продовження тренування: python resume_training.py")

if __name__ == "__main__":
    main()