import urllib.request
import ssl
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Обхід SSL перевірки
ssl._create_default_https_context = ssl._create_unverified_context

class DownloadProgressBar(tqdm):
    """Progress bar для urllib"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    """Завантажує файл з progress bar"""
    print(f"Завантаження: {output_path.name}")
    print(f"Джерело: {url}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='Прогрес') as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    
    print(f"✓ Завантажено: {output_path.name}\n")

def verify_data(npz_path):
    """Перевіряє завантажені дані"""
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        features = data['data']
        pdbids = data['pdbids']
        
        print(f"✓ Дані завантажені успішно!")
        print(f"  Форма features: {features.shape}")
        print(f"  Кількість білків: {len(pdbids)}")
        print(f"  Розмір файлу: {npz_path.stat().st_size / (1024**2):.2f} MB")
        
        # Інформація про перший білок
        first_protein = features[0]
        seq_mask = first_protein[:, 50]  # Маска послідовності
        seq_len = int(seq_mask.sum())
        
        print(f"  Приклад (перший білок):")
        print(f"    PDB ID: {pdbids[0]}")
        print(f"    Довжина: {seq_len} залишків")
        
        return True
        
    except Exception as e:
        print(f"✗ Помилка при перевірці: {e}")
        return False

def main():
    print("="*70)
    print("ЗАВАНТАЖЕННЯ ДАТАСЕТУ NetSurfP-3.0")
    print("="*70)
    print()
    print("Датасет від DTU Health Tech (Технічний університет Данії)")
    print("Офіційний benchmark для прогнозування вторинної структури білків")
    print()
    
    # Створюємо папку для даних
    data_dir = Path("data/raw/netsurf")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs для завантаження
    datasets = {
        'Train_HHblits.npz': 'https://services.healthtech.dtu.dk/services/NetSurfP-3.0/training_data/Train_HHblits.npz',
        'CB513_HHblits.npz': 'https://services.healthtech.dtu.dk/services/NetSurfP-3.0/training_data/CB513_HHblits.npz',
        'TS115_HHblits.npz': 'https://services.healthtech.dtu.dk/services/NetSurfP-3.0/training_data/TS115_HHblits.npz',
    }
    
    print("Буде завантажено 3 файли:")
    print("  1. Train_HHblits.npz  - Тренувальний набір (~150 MB)")
    print("  2. CB513_HHblits.npz  - Тестовий набір CB513 (~5 MB)")
    print("  3. TS115_HHblits.npz  - Тестовий набір TS115 (~1 MB)")
    print()
    
    downloaded = []
    
    for filename, url in datasets.items():
        output_path = data_dir / filename
        
        # Перевірка чи файл вже існує
        if output_path.exists():
            print(f"✓ {filename} вже існує, пропускаємо...")
            if verify_data(output_path):
                downloaded.append(filename)
            print()
            continue
        
        # Завантаження
        try:
            download_file(url, output_path)
            
            # Перевірка
            if verify_data(output_path):
                downloaded.append(filename)
            else:
                print(f"⚠️  Файл завантажився, але містить помилки")
            
        except Exception as e:
            print(f"✗ Помилка при завантаженні {filename}: {e}")
            print()
            continue
        
        print()
    
    # Підсумок
    print("="*70)
    if len(downloaded) >= 2:  # Мінімум Train + один тестовий
        print("✓ ЗАВАНТАЖЕННЯ ЗАВЕРШЕНО УСПІШНО!")
        print("="*70)
        print()
        print(f"Завантажено {len(downloaded)}/3 файлів:")
        for f in downloaded:
            print(f"  ✓ {f}")
        print()
        print("Наступні кроки:")
        print("  1. Запустіть: python scripts\\prepare_data.py")
        print("  2. Або відкрийте: notebooks\\02_netsurf_exploration.ipynb")
    else:
        print("⚠️  ЗАВАНТАЖЕННЯ НЕПОВНЕ")
        print("="*70)
        print()
        print("Деякі файли не завантажилися.")
        print("Можливі причини:")
        print("  - Відсутнє інтернет-з'єднання")
        print("  - Сервер DTU тимчасово недоступний")
        print("  - Firewall блокує доступ")
        print()
        print("Спробуйте запустити скрипт знову пізніше.")

if __name__ == "__main__":
    main()
