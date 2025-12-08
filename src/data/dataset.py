"""
PyTorch Dataset для NetSurfP-3.0 датасету
Підтримує Q3 та Q8 класифікацію вторинної структури
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional

class NetSurfPDataset(Dataset):
    """
    Dataset для NetSurfP-3.0 формату (.npz файли)
    
    Структура даних на кожен резидуум (68 features):
        - 0:20   - One-hot амінокислоти
        - 20:50  - HMM профіль (30 dim)
        - 50     - Маска послідовності (1=валідний, 0=padding)
        - 57:65  - Q8 мітки (one-hot)
    
    Параметри:
        npz_path: шлях до .npz файлу
        max_len: максимальна довжина послідовності (для padding)
        use_q3: якщо True, конвертує Q8 → Q3
    """
    
    # Маппінг Q8 → Q3
    # G, H, I (спіралі) → H (helix)
    # E, B (листи) → E (sheet)  
    # T, S, C (решта) → C (coil)
    Q8_TO_Q3 = {
        0: 0,  # G → H
        1: 0,  # H → H
        2: 0,  # I → H
        3: 1,  # E → E
        4: 1,  # B → E
        5: 2,  # T → C
        6: 2,  # S → C
        7: 2   # C → C
    }
    
    def __init__(
        self, 
        npz_path: str,
        max_len: int = 500,
        use_q3: bool = True
    ):
        print(f"Завантаження датасету з {npz_path}...")
        
        # Завантаження .npz файлу
        data = np.load(npz_path, allow_pickle=True)
        
        self.features = data['data']      # Масив білків
        self.pdbids = data['pdbids']      # PDB ідентифікатори
        self.max_len = max_len
        self.use_q3 = use_q3
        self.num_classes = 3 if use_q3 else 8
        
        print(f"✓ Завантажено {len(self.pdbids)} білків")
        print(f"  Режим: {'Q3' if use_q3 else 'Q8'} класифікація")
        print(f"  Max length: {max_len}")
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Повертає один білок
        
        Returns:
            dict з ключами:
                - sequence: torch.Tensor [max_len, 50] - features
                - label: torch.Tensor [max_len] - Q3 або Q8 індекси
                - length: int - реальна довжина
                - pdb_id: str - ідентифікатор білка
        """
        protein = self.features[idx]
        
        # Витягуємо компоненти
        amino_acids = protein[:, 0:20]    # One-hot амінокислот
        hmm_profile = protein[:, 20:50]   # HMM профіль
        mask = protein[:, 50]             # Маска послідовності
        q8_labels = protein[:, 57:65]     # Q8 мітки
        
        # Знаходимо реальну довжину послідовності
        seq_len = int(mask.sum())
        
        # Обрізаємо до реальної довжини (без padding)
        amino_acids = amino_acids[:seq_len]
        hmm_profile = hmm_profile[:seq_len]
        q8_labels = q8_labels[:seq_len]
        
        # Об'єднуємо features: AA + HMM = 50 features
        X = np.concatenate([amino_acids, hmm_profile], axis=-1)
        
        # Конвертуємо Q8 one-hot → індекси класів
        y = np.argmax(q8_labels, axis=-1)
        
        # Q8 → Q3 конвертація якщо потрібно
        if self.use_q3:
            y = np.array([self.Q8_TO_Q3[label] for label in y])
        
        # Padding до max_len
        if seq_len > self.max_len:
            X = X[:self.max_len]
            y = y[:self.max_len]
            seq_len = self.max_len
        
        # Створюємо padded масиви
        X_padded = np.zeros((self.max_len, 50), dtype=np.float32)
        y_padded = np.full(self.max_len, -100, dtype=np.int64)  # -100 = ignore_index
        
        X_padded[:seq_len] = X
        y_padded[:seq_len] = y
        
        return {
            'sequence': torch.from_numpy(X_padded),
            'label': torch.from_numpy(y_padded),
            'length': seq_len,
            'pdb_id': self.pdbids[idx]
        }


def create_dataloaders(
    train_path: str,
    test_path: str,
    batch_size: int = 32,
    max_len: int = 500,
    use_q3: bool = True,
    num_workers: int = 0
):
    """
    Створює DataLoader для train та test
    
    Параметри:
        train_path: шлях до Train_HHblits.npz
        test_path: шлях до CB513_HHblits.npz або TS115_HHblits.npz
        batch_size: розмір батчу
        max_len: максимальна довжина
        use_q3: Q3 або Q8 класифікація
        num_workers: ЗАВЖДИ 0 для Windows!
    
    Returns:
        train_loader, test_loader
    """
    print("Створення DataLoaders...")
    
    # Створення Dataset
    train_dataset = NetSurfPDataset(train_path, max_len, use_q3)
    test_dataset = NetSurfPDataset(test_path, max_len, use_q3)
    
    # Створення DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # False для CPU
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"✓ DataLoaders створено:")
    print(f"  Train: {len(train_dataset)} білків, {len(train_loader)} батчів")
    print(f"  Test:  {len(test_dataset)} білків, {len(test_loader)} батчів")
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Тестування Dataset
    print("=== ТЕСТ DATASET ===\n")
    
    dataset = NetSurfPDataset(
        "data/raw/netsurf/Train_HHblits.npz",
        max_len=500,
        use_q3=True
    )
    
    # Перший білок
    sample = dataset[0]
    
    print(f"\nПриклад даних (білок #{0}):")
    print(f"  PDB ID: {sample['pdb_id']}")
    print(f"  Довжина: {sample['length']}")
    print(f"  Форма sequence: {sample['sequence'].shape}")
    print(f"  Форма label: {sample['label'].shape}")
    print(f"  Мітки (перші 20): {sample['label'][:20]}")
    
    # Статистика класів
    labels = sample['label'][:sample['length']]
    unique, counts = torch.unique(labels, return_counts=True)
    
    print(f"\n  Розподіл класів у цьому білку:")
    class_names = ['H (Helix)', 'E (Sheet)', 'C (Coil)']
    for cls, count in zip(unique, counts):
        if cls != -100:
            print(f"    {class_names[cls]}: {count} ({count/sample['length']*100:.1f}%)")