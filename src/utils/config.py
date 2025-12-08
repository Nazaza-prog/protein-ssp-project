"""
Утиліти для конфігурації та оптимізації
"""
import os
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime

def setup_cpu_optimization(num_threads: int = 12):
    """
    Налаштовує CPU для максимальної продуктивності
    
    Параметри:
        num_threads: кількість потоків (фізичні ядра процесора)
    """
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    torch.set_num_threads(num_threads)
    
    # set_num_interop_threads можна викликати тільки один раз
    # Перевіряємо чи вже викликано
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        # Вже встановлено, ігноруємо
        pass
    
    print(f"✓ CPU оптимізація: {num_threads} потоків")

def set_seed(seed: int = 42):
    """
    Встановлює seed для відтворюваності результатів
    
    Параметри:
        seed: значення seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Для повної детермінованості (повільніше)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed: {seed}")

def get_device(prefer_cuda: bool = True):
    """
    Визначає найкращий доступний device
    
    Параметри:
        prefer_cuda: чи використовувати CUDA якщо доступна
        
    Повертає:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"✓ Device: CPU")
        if prefer_cuda:
            print("  ⚠️  CUDA недоступна, використовується CPU")
    
    return device

def safe_save_checkpoint(checkpoint_data, save_path, backup=True):
    """
    Безпечне збереження checkpoint з backup попередньої версії
    
    Параметри:
        checkpoint_data: dict з даними для збереження
        save_path: Path або str - шлях до файлу
        backup: bool - чи створювати backup попередньої версії
    
    Повертає:
        bool: True якщо збереження успішне
    
    Приклад використання:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'config': config
        }
        
        success = safe_save_checkpoint(
            checkpoint, 
            save_dir / 'last_epoch.pt',
            backup=True
        )
    """
    save_path = Path(save_path)
    
    try:
        # Додаємо timestamp до checkpoint
        checkpoint_data['saved_at'] = datetime.now().isoformat()
        
        # Якщо файл існує і потрібен backup
        if backup and save_path.exists():
            backup_path = save_path.parent / f"{save_path.stem}_backup{save_path.suffix}"
            
            # Зберігаємо попередню версію як backup
            if backup_path.exists():
                backup_path.unlink()  # Видаляємо старий backup
            
            save_path.rename(backup_path)
        
        # Зберігаємо новий checkpoint
        torch.save(checkpoint_data, save_path)
        
        # Перевіряємо що файл створено і має розумний розмір
        if save_path.exists() and save_path.stat().st_size > 1000:
            return True
        else:
            print(f"⚠️  Попередження: checkpoint може бути пошкоджений ({save_path})")
            return False
            
    except Exception as e:
        print(f"✗ Помилка при збереженні checkpoint: {e}")
        return False

def load_checkpoint_safe(checkpoint_path, backup_on_error=True):
    """
    Безпечне завантаження checkpoint з автоматичним fallback на backup
    
    Параметри:
        checkpoint_path: Path або str - шлях до checkpoint
        backup_on_error: bool - чи пробувати backup якщо основний пошкоджений
    
    Повертає:
        dict або None: завантажений checkpoint або None при помилці
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint не знайдено: {checkpoint_path}")
        return None
    
    try:
        # Пробуємо завантажити основний checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Checkpoint завантажено: {checkpoint_path.name}")
        
        # Показуємо інформацію
        if 'saved_at' in checkpoint:
            print(f"  Збережено: {checkpoint['saved_at']}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'best_val_acc' in checkpoint:
            print(f"  Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
        
        return checkpoint
        
    except Exception as e:
        print(f"✗ Помилка при завантаженні {checkpoint_path.name}: {e}")
        
        # Пробуємо backup
        if backup_on_error:
            backup_path = checkpoint_path.parent / f"{checkpoint_path.stem}_backup{checkpoint_path.suffix}"
            
            if backup_path.exists():
                print(f"  Спроба завантажити backup: {backup_path.name}")
                try:
                    checkpoint = torch.load(backup_path, map_location='cpu')
                    print(f"  ✓ Backup checkpoint завантажено успішно!")
                    
                    if 'epoch' in checkpoint:
                        print(f"  Epoch: {checkpoint['epoch']}")
                    
                    return checkpoint
                    
                except Exception as backup_error:
                    print(f"  ✗ Backup також пошкоджений: {backup_error}")
        
        return None

def print_system_info():
    """Друкує інформацію про систему"""
    import platform
    import psutil
    
    print("\n" + "="*70)
    print("ІНФОРМАЦІЯ ПРО СИСТЕМУ")
    print("="*70)
    
    # OS
    print(f"ОС: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # CPU
    print(f"CPU: {platform.processor()}")
    print(f"  Фізичні ядра: {psutil.cpu_count(logical=False)}")
    print(f"  Логічні ядра: {psutil.cpu_count(logical=True)}")
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.total / (1024**3):.1f} GB")
    print(f"  Доступно: {ram.available / (1024**3):.1f} GB")
    
    # PyTorch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    print("="*70 + "\n")