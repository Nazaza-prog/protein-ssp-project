"""
Продовження тренування з checkpoint
Підтримка автоматичного fallback на backup при пошкодженні

Розташування: src/training/resume_training.py
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

from src.data.dataset import create_dataloaders
from src.models.cnn_lstm_model import create_model
from src.evaluation.metrics import MetricsTracker
from src.utils.config import (
    setup_cpu_optimization,
    set_seed,
    get_device,
    safe_save_checkpoint,
    load_checkpoint_safe
)

# Глобальна змінна для graceful shutdown
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

# Реєструємо обробник
signal.signal(signal.SIGINT, signal_handler)

# CPU оптимізації
setup_cpu_optimization(num_threads=12)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Один epoch тренування"""
    model.train()
    
    total_loss = 0
    metrics = MetricsTracker(num_classes=3)
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch in pbar:
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(sequences)
        
        # Loss
        loss = criterion(outputs.view(-1, 3), labels.view(-1))
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Метрики
        total_loss += loss.item()
        metrics.update(outputs.detach(), labels)
        
        # Оновлюємо progress bar
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
        file_path = save_dir / 'last_epoch.pt'
        success = safe_save_checkpoint(checkpoint_data, file_path, backup=True)
        if success:
            print(f"  💾 Автозбереження: last_epoch.pt (epoch {epoch})")
    
    elif checkpoint_type == 'best':
        file_path = save_dir / 'best_model.pt'
        checkpoint_data['val_acc'] = checkpoint_data['best_val_acc']
        success = safe_save_checkpoint(checkpoint_data, file_path, backup=False)
        if success:
            print(f"  ✓ Збережено найкращу модель (Q3: {best_val_acc:.2f}%)")
    
    elif checkpoint_type == 'archive':
        file_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
        success = safe_save_checkpoint(checkpoint_data, file_path, backup=False)
        if success:
            print(f"  📦 Архівне збереження: checkpoint_epoch_{epoch}.pt")
    
    elif checkpoint_type == 'interrupted':
        checkpoint_data['stopped_manually'] = True
        file_path = save_dir / 'interrupted_checkpoint.pt'
        success = safe_save_checkpoint(checkpoint_data, file_path, backup=False)
        if success:
            print(f"  ✓ Checkpoint збережено: interrupted_checkpoint.pt")
    
    return success

def find_latest_checkpoint(results_dir):
    """
    Знаходить найновіший checkpoint для продовження
    
    Повертає:
        Path або None: шлях до checkpoint або None
    """
    results_dir = Path(results_dir)
    
    # Пріоритет checkpoint'ів
    checkpoint_candidates = [
        ('last_epoch.pt', 'Остання епоха (автозбереження)'),
        ('interrupted_checkpoint.pt', 'Переривання (Ctrl+C)'),
        ('final_checkpoint.pt', 'Фінальний (Early Stopping)'),
        ('best_model.pt', 'Найкраща модель'),
    ]
    
    print("\n🔍 Пошук доступних checkpoint...")
    
    for filename, description in checkpoint_candidates:
        checkpoint_path = results_dir / filename
        if checkpoint_path.exists():
            print(f"✓ Знайдено: {filename} ({description})")
            
            # Перевіряємо чи є backup
            backup_path = results_dir / f"{checkpoint_path.stem}_backup{checkpoint_path.suffix}"
            if backup_path.exists():
                print(f"  └─ Backup: {backup_path.name}")
            
            return checkpoint_path
    
    # Також шукаємо архівні checkpoint (checkpoint_epoch_X.pt)
    if results_dir.exists():
        epoch_checkpoints = sorted(
            results_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1]),
            reverse=True
        )
        
        if epoch_checkpoints:
            latest = epoch_checkpoints[0]
            print(f"✓ Знайдено: {latest.name} (Архівний checkpoint)")
            return latest
    
    return None

def main():
    print("\n" + "="*70)
    print("ПРОДОВЖЕННЯ ТРЕНУВАННЯ")
    print("="*70 + "\n")
    
    results_dir = Path("results/baseline")
    
    # Знаходимо найновіший checkpoint
    checkpoint_path = find_latest_checkpoint(results_dir)
    
    if checkpoint_path is None:
        print(f"\n✗ Жоден checkpoint не знайдено в {results_dir}")
        print("\nДоступні файли:")
        if results_dir.exists():
            pt_files = list(results_dir.glob("*.pt"))
            if pt_files:
                for f in pt_files:
                    print(f"  - {f.name}")
            else:
                print("  (жодного .pt файлу не знайдено)")
        else:
            print("  (папка не існує)")
        print("\nСпочатку запустіть: python train.py")
        return
    
    print(f"\n📁 Використовується checkpoint: {checkpoint_path.name}")
    print()
    
    # Завантаження checkpoint з автоматичним fallback на backup
    checkpoint = load_checkpoint_safe(checkpoint_path, backup_on_error=True)
    
    if checkpoint is None:
        print("\n✗ Не вдалося завантажити жоден checkpoint")
        print("Можливо файли пошкоджені. Спробуйте:")
        print("  1. Видалити пошкоджені файли")
        print("  2. Запустити train.py заново")
        return
    
    config = checkpoint['config']
    start_epoch = checkpoint['epoch'] + 1
    best_val_acc = checkpoint.get('best_val_acc', 0)
    
    # Перевірка чи можна продовжувати
    if start_epoch > config['training']['num_epochs']:
        print(f"\n⚠️  Увага: Останній epoch ({checkpoint['epoch']}) >= max epochs ({config['training']['num_epochs']})")
        print("\nВиберіть дію:")
        print(f"  1. Продовжити з epoch {start_epoch}")
        print(f"  2. Змінити num_epochs в конфігурації")
        print(f"  3. Завершити")
        return
    
    print(f"\n▶️  Продовження з епохи {start_epoch}")
    print()
    
    # Device
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("✓ Ваги моделі відновлено")
    
    # Optimizer
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
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✓ Стан optimizer відновлено")
    
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
    
    print("\n" + "="*70)
    print("ПРОДОВЖЕННЯ ТРЕНУВАННЯ")
    print("="*70 + "\n")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        # Перевірка graceful shutdown
        global stop_training
        if stop_training:
            print("\n" + "="*70)
            print("ТРЕНУВАННЯ ЗУПИНЕНО КОРИСТУВАЧЕМ")
            print("="*70)
            
            # Зберігаємо поточний стан
            save_dir = root_dir / config['logging']['log_dir']
            save_dir.mkdir(parents=True, exist_ok=True)
            
            save_checkpoint_safely(
                epoch - 1, model, optimizer, best_val_acc,
                config, save_dir, checkpoint_type='interrupted'
            )
            
            print(f"\nОстанній завершений epoch: {epoch - 1}")
            print(f"Найкраща валідаційна точність: {best_val_acc:.2f}%")
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
        
        # Збереження
        print(f"\n  💾 Збереження checkpoint...")
        save_dir = root_dir / config['logging']['log_dir']
        
        # 1. Автозбереження ЗАВЖДИ
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
        
        # Early stopping
        if config['training']['early_stopping']['use']:
            if patience_counter >= config['training']['early_stopping']['patience']:
                print(f"\n⚠️  Early stopping після {epoch} епох")
                print(f"Найкраща валідаційна точність: {best_val_acc:.2f}%")
                
                # Завантажуємо найкращу модель
                best_checkpoint = torch.load(save_dir / 'best_model.pt')
                
                # Зберігаємо final checkpoint
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
    
    print("\n" + "="*70)
    print("✓ ТРЕНУВАННЯ ЗАВЕРШЕНО!")
    print("="*70)
    print(f"\n🎯 Найкраща точність: Q3 = {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()