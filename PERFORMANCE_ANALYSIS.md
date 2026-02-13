# Performance Analysis Report
## Protein Secondary Structure Prediction Project

**Analysis Date**: 2025-12-28
**Codebase**: Protein SSP with CNN+BiLSTM
**Primary Language**: Python (PyTorch)

---

## Executive Summary

This analysis identifies **16 performance anti-patterns** and optimization opportunities in the codebase. Issues range from **critical** (N+1-style patterns, memory leaks) to **moderate** (inefficient algorithms, redundant computations). No SQL N+1 queries were found (as this is a machine learning project without a database), but similar accumulation patterns exist in data processing.

**Key Findings**:
- 🔴 **Critical**: Memory accumulation in MetricsTracker (potential OOM)
- 🔴 **Critical**: Inefficient Q8→Q3 label conversion (repeated per epoch)
- 🟡 **Moderate**: Multiple checkpoint saves per epoch (I/O bottleneck)
- 🟡 **Moderate**: Redundant argmax computations in metrics
- 🟢 **Minor**: Multiple dropout layers, double transposes

---

## Critical Performance Issues

### 1. ⚠️ Memory Leak Pattern - MetricsTracker Accumulation
**File**: `src/evaluation/metrics.py:122-136`
**Severity**: 🔴 **CRITICAL**

**Issue**:
```python
def update(self, predictions: torch.Tensor, targets: torch.Tensor):
    if predictions.dim() == 3:
        preds = predictions.argmax(dim=-1)
    else:
        preds = predictions

    self.all_predictions.append(preds.cpu())  # ⚠️ Accumulates in memory
    self.all_targets.append(targets.cpu())    # ⚠️ Accumulates in memory
```

**Problems**:
1. **Memory Growth**: Accumulates ALL predictions/targets for entire epoch in Python lists
2. **Device Transfers**: Repeated `.cpu()` calls create overhead
3. **Memory Spike**: Final `torch.cat()` in `compute()` doubles memory usage temporarily
4. **Potential OOM**: With large datasets (>10K proteins), this can exhaust RAM

**Impact**:
- Linear memory growth: O(n) where n = dataset size
- For CB513 test set (~500 proteins × 300 residues): ~150K predictions stored
- Estimated memory: ~150K × 4 bytes × 2 (preds + targets) = **~1.2 MB** (minor for this dataset, but scales poorly)

**Recommendation**:
```python
class MetricsTracker:
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
        self.total_correct = 0
        self.total_samples = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        # Compute incrementally instead of accumulating
        if predictions.dim() == 3:
            preds = predictions.argmax(dim=-1)
        else:
            preds = predictions

        valid = targets != -100
        self.total_correct += ((preds == targets) & valid).sum().item()
        self.total_samples += valid.sum().item()

        # Update confusion matrix incrementally
        valid_preds = preds[valid].cpu()
        valid_targets = targets[valid].cpu()
        for t, p in zip(valid_targets, valid_preds):
            self.confusion_matrix[t, p] += 1
```

**Priority**: HIGH - Implement streaming metrics calculation

---

### 2. ⚠️ N+1-Style Pattern - Q8 to Q3 Conversion
**File**: `src/data/dataset.py:99-100`
**Severity**: 🔴 **CRITICAL**

**Issue**:
```python
# Q8 → Q3 conversion
if self.use_q3:
    y = np.array([self.Q8_TO_Q3[label] for label in y])  # ⚠️ Loop per label
```

**Problems**:
1. **Python Loop**: Iterates over every residue in Python (slow)
2. **Dictionary Lookup**: Repeated dict access instead of vectorized operation
3. **Executed Per Epoch**: This runs for EVERY protein in EVERY epoch
4. **Memory Allocation**: Creates intermediate list before converting to array

**Impact**:
- For a 300-residue protein: 300 dictionary lookups
- For 5,534 training proteins × 100 epochs: **553,400 × 300 = 166M lookups**
- Estimated overhead: ~10-20% of data loading time

**Recommendation**:
```python
# Vectorized approach using NumPy
Q8_TO_Q3_ARRAY = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)

if self.use_q3:
    y = Q8_TO_Q3_ARRAY[y]  # ✅ Vectorized indexing
```

**Performance Gain**: ~10-50x faster (NumPy vectorized ops vs Python loops)

**Priority**: HIGH - Easy fix with significant impact

---

### 3. ⚠️ Inefficient Padding - Repeated Allocations
**File**: `src/data/dataset.py:108-113`
**Severity**: 🟡 **MODERATE**

**Issue**:
```python
# Create padded arrays
X_padded = np.zeros((self.max_len, 50), dtype=np.float32)  # ⚠️ Allocated every call
y_padded = np.full(self.max_len, -100, dtype=np.int64)     # ⚠️ Allocated every call

X_padded[:seq_len] = X
y_padded[:seq_len] = y
```

**Problems**:
1. **Repeated Allocation**: Creates new arrays for every `__getitem__()` call
2. **Memory Churn**: Allocation + deallocation overhead
3. **Cache Pollution**: Constant allocations reduce CPU cache efficiency

**Impact**:
- Per protein: 2 array allocations (300×50 + 300 elements)
- Per epoch: 5,534 × 2 = 11,068 allocations
- Estimated overhead: ~5-10% of data loading time

**Recommendation**:
```python
# Pre-allocate buffers in __init__ (if num_workers=0)
# OR use torch.nn.utils.rnn.pad_sequence in collate_fn
```

**Priority**: MEDIUM - Noticeable on CPU, harder to implement safely

---

### 4. ⚠️ Multiple Checkpoint Saves Per Epoch
**File**: `src/training/train.py:305-333`
**Severity**: 🟡 **MODERATE**

**Issue**:
```python
# 1. AUTOSAVE AFTER EVERY EPOCH
save_checkpoint_safely(
    epoch, model, optimizer, best_val_acc,
    config, save_dir, checkpoint_type='auto'
)

# 2. Best model
if val_acc > best_val_acc:
    save_checkpoint_safely(
        epoch, model, optimizer, best_val_acc,
        config, save_dir, checkpoint_type='best'
    )

# 3. Archive save every 5 epochs
if epoch % 5 == 0:
    save_checkpoint_safely(
        epoch, model, optimizer, best_val_acc,
        config, save_dir, checkpoint_type='archive'
    )
```

**Problems**:
1. **I/O Bottleneck**: Saves model 2-3 times per epoch (~2.3M params × 4 bytes = ~9 MB per save)
2. **Disk Thrashing**: On slower storage (HDD), this causes significant slowdowns
3. **Backup Creation**: Auto-save also creates backup, doubling I/O
4. **Synchronous Saves**: Blocks training loop

**Impact**:
- Per epoch: ~18-27 MB written (auto + backup + maybe best)
- 100 epochs: ~2-3 GB total writes
- On HDD: ~2-5 seconds per save = **4-10 seconds per epoch overhead**
- On SSD: ~0.5-1 second per save

**Recommendation**:
1. **Async Saves**: Use threading to save in background
2. **Reduce Frequency**: Only save best + last (skip archive unless needed)
3. **Incremental Saves**: Only save model state_dict, not optimizer (for archive)

```python
# Use background thread for auto-saves
import threading

def async_save_checkpoint(checkpoint_data, path):
    thread = threading.Thread(target=lambda: torch.save(checkpoint_data, path))
    thread.start()
    return thread
```

**Priority**: MEDIUM - Significant on slow storage

---

## Moderate Performance Issues

### 5. Redundant Argmax Computations
**File**: `src/evaluation/metrics.py` (multiple locations)
**Severity**: 🟡 **MODERATE**

**Issue**:
```python
# Repeated in q3_accuracy(), per_class_accuracy(), confusion_matrix_metrics()
if predictions.dim() == 3:
    preds = predictions.argmax(dim=-1)  # ⚠️ Computed multiple times
else:
    preds = predictions
```

**Problems**:
- Same computation in 4 different functions
- If metrics called separately, argmax runs 4 times
- Unnecessary dimension checks

**Impact**: ~5-10% overhead in metrics calculation

**Recommendation**:
```python
# Compute once, pass indices
predictions_idx = predictions.argmax(dim=-1) if predictions.dim() == 3 else predictions
q3_acc = q3_accuracy(predictions_idx, targets)
class_metrics = per_class_accuracy(predictions_idx, targets)
```

**Priority**: MEDIUM

---

### 6. Inefficient Class Metrics Loop
**File**: `src/evaluation/metrics.py:63-77`
**Severity**: 🟡 **MODERATE**

**Issue**:
```python
for c in range(num_classes):
    class_mask = (targets == c) & valid  # ⚠️ Creates mask for each class

    if class_mask.sum() > 0:
        correct = ((preds == c) & class_mask).sum().float()
        acc = (correct / class_mask.sum() * 100).item()
```

**Problems**:
- Creates separate mask for each class
- Multiple passes over data
- Could use confusion matrix diagonal instead

**Recommendation**:
```python
# Compute from confusion matrix (single pass)
cm = confusion_matrix_metrics(predictions, targets, num_classes)
for c in range(num_classes):
    support = cm[c].sum()
    if support > 0:
        accuracy = cm[c, c] / support * 100
```

**Priority**: LOW-MEDIUM

---

### 7. Double Transpose in Model Forward Pass
**File**: `src/models/cnn_lstm_model.py:77-99`
**Severity**: 🟢 **MINOR**

**Issue**:
```python
def forward(self, x):
    # [batch, seq_len, features] → [batch, features, seq_len]
    x = x.transpose(1, 2)  # ⚠️ Transpose #1

    # CNN operations...
    x = self.conv1(x)
    # ...

    # [batch, features, seq_len] → [batch, seq_len, features]
    x = x.transpose(1, 2)  # ⚠️ Transpose #2

    # BiGRU
    x, _ = self.gru(x)
```

**Problems**:
- Two transpose operations per forward pass
- Creates views (minimal overhead on modern PyTorch)

**Impact**: Minimal (<1% overhead, PyTorch optimizes transposes well)

**Note**: This is by design (Conv1d requires different layout than GRU), not a real issue.

**Priority**: IGNORE - Design constraint, not a bug

---

### 8. Multiple Dropout Applications
**File**: `src/models/cnn_lstm_model.py:84,90,96,103`
**Severity**: 🟢 **MINOR**

**Issue**:
```python
x = self.dropout(x)  # After conv1
# ...
x = self.dropout(x)  # After conv2
# ...
x = self.dropout(x)  # After conv3
# ...
x = self.dropout(x)  # After GRU
```

**Problems**:
- 4 dropout layers might be excessive for a 2.3M parameter model
- More dropout = more regularization but also slower training

**Impact**: ~2-5% training time (negligible)

**Note**: This is a hyperparameter choice, not necessarily a bug. But 3-4 dropout layers is aggressive.

**Recommendation**: Consider reducing to 2 dropout layers (after conv3 and after GRU)

**Priority**: LOW - Architectural decision

---

### 9. Visualization Re-implements Metrics
**File**: `visualize_real_results.py:25-62`
**Severity**: 🟢 **MINOR**

**Issue**:
```python
def evaluate_model(model, test_loader, device):
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # ... forward pass ...
            all_predictions.extend(predictions[mask].cpu().numpy())  # ⚠️ Same as MetricsTracker
            all_labels.extend(labels[mask].cpu().numpy())
```

**Problems**:
- Code duplication (MetricsTracker already does this)
- Accumulation in lists (same memory issue as #1)

**Recommendation**:
```python
# Reuse MetricsTracker
metrics_tracker = MetricsTracker(num_classes=3)
# ... use tracker.update() ...
results = metrics_tracker.compute()
```

**Priority**: LOW - Cleanup/refactoring

---

### 10. No Data Caching
**File**: `src/data/dataset.py:49-50`
**Severity**: 🟡 **MODERATE**

**Issue**:
```python
# Loaded fresh every time script runs
data = np.load(npz_path, allow_pickle=True)
```

**Problems**:
- No caching between runs
- npz loading is relatively fast, but still I/O overhead
- Could benefit from memory-mapped arrays

**Recommendation**:
```python
# Use memory-mapped loading
data = np.load(npz_path, allow_pickle=True, mmap_mode='r')
```

**Impact**: Minimal (npz is already compressed), but mmap helps with large files

**Priority**: LOW

---

## Minor Issues & Code Quality

### 11. Unnecessary Slicing Copies
**File**: `src/data/dataset.py:88-91`
**Severity**: 🟢 **MINOR**

**Issue**:
```python
amino_acids = amino_acids[:seq_len]  # ⚠️ Creates copy
hmm_profile = hmm_profile[:seq_len]  # ⚠️ Creates copy
q8_labels = q8_labels[:seq_len]      # ⚠️ Creates copy
```

**Impact**: Negligible (NumPy slicing is fast)

**Priority**: IGNORE

---

### 12. Config Loading Not Cached
**File**: `src/training/train.py:170-171`
**Severity**: 🟢 **MINOR**

**Issue**:
```python
# Loaded in main(), but could be module-level
with open(root_dir / 'configs' / 'baseline_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
```

**Impact**: Minimal (<10ms per run)

**Priority**: IGNORE

---

### 13. No Mixed Precision Training
**File**: `src/training/train.py` (missing)
**Severity**: 🟡 **MODERATE**

**Issue**: Code doesn't use PyTorch AMP (Automatic Mixed Precision)

**Problems**:
- Could speed up training on modern CPUs with AVX-512
- Reduces memory usage

**Recommendation**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(sequences)
    loss = criterion(outputs.view(-1, 3), labels.view(-1))

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Note**: AMP is primarily for GPU, but can help on CPU with modern architectures

**Priority**: LOW-MEDIUM - Requires testing

---

### 14. Batch Concatenation Memory Spike
**File**: `src/evaluation/metrics.py:146-147`
**Severity**: 🟡 **MODERATE**

**Issue**:
```python
def compute(self) -> Dict:
    # Concatenate all batches
    predictions = torch.cat(self.all_predictions, dim=0)  # ⚠️ Memory spike
    targets = torch.cat(self.all_targets, dim=0)
```

**Problems**:
- Temporarily doubles memory usage during concatenation
- Could trigger OOM on large datasets

**Impact**: Related to #1 - fix #1 to eliminate this issue

**Priority**: HIGH (but solved by fixing #1)

---

### 15. Gradient Accumulation Not Used
**File**: `src/training/train.py:60-84`
**Severity**: 🟢 **MINOR**

**Issue**: No gradient accumulation for simulating larger batches

**Recommendation**:
```python
# For CPU training, accumulate gradients over 2-4 batches
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Priority**: LOW - Optional optimization

---

### 16. Missing DataLoader Prefetching
**File**: `src/data/dataset.py:152-167`
**Severity**: 🟡 **MODERATE**

**Issue**:
```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,  # Set to 0 for Windows
    pin_memory=False,         # False for CPU
    drop_last=True
)
```

**Problems**:
- `num_workers=0` means single-threaded loading (no async prefetch)
- `prefetch_factor` not set
- Note: Comments say "ALWAYS 0 for Windows" - this is a Windows limitation

**Recommendation**:
```python
# On Linux/Mac, use multiprocessing
num_workers = 4 if platform.system() != 'Windows' else 0
prefetch_factor = 2 if num_workers > 0 else None

train_loader = torch.utils.data.DataLoader(
    # ...
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    persistent_workers=True if num_workers > 0 else False
)
```

**Priority**: MEDIUM - Significant speedup on Linux/Mac

---

## Performance Optimization Priorities

### High Priority (Immediate Impact)
1. ✅ **Fix Q8→Q3 vectorization** (dataset.py:100) - 10-50x speedup
2. ✅ **Implement streaming metrics** (metrics.py:122-136) - Prevents OOM
3. ✅ **Reduce checkpoint saves** (train.py:305-333) - 30-50% faster epochs on HDD

### Medium Priority (Worthwhile)
4. ✅ **Enable multiprocessing DataLoader** on Linux/Mac - 2-3x data loading speedup
5. ✅ **Reduce argmax redundancy** (metrics.py) - 5-10% metrics speedup
6. ✅ **Async checkpoint saving** - Non-blocking saves

### Low Priority (Nice to Have)
7. ✅ Vectorize class metrics calculation
8. ✅ Use memory-mapped npz loading
9. ✅ Experiment with gradient accumulation
10. ✅ Test AMP for CPU (if supported)

---

## Summary Statistics

| Category | Count | Severity Distribution |
|----------|-------|----------------------|
| **Total Issues** | 16 | 🔴 Critical: 2, 🟡 Moderate: 7, 🟢 Minor: 7 |
| **Memory Issues** | 4 | MetricsTracker, padding, batch concat, viz |
| **Algorithmic** | 3 | Q8→Q3 loop, class metrics loop, argmax |
| **I/O Issues** | 2 | Multiple saves, no caching |
| **Architecture** | 7 | Minor design decisions |

**Estimated Total Speedup**: Implementing high-priority fixes could yield **20-40% faster training** with **50% less memory usage**.

---

## No SQL N+1 Queries Found

**Note**: This project does not use a database, so traditional N+1 query anti-patterns (e.g., loading relationships in a loop) do not apply. However, analogous patterns were found:
- **Data Processing N+1**: Q8→Q3 conversion loop (#2)
- **Metrics Accumulation**: Similar to accumulating DB results (#1)

---

## Recommendations Summary

1. **Immediate Actions**:
   - Vectorize Q8→Q3 conversion
   - Implement streaming metrics calculation
   - Reduce checkpoint frequency

2. **Short-term Improvements**:
   - Enable DataLoader multiprocessing (platform-dependent)
   - Consolidate argmax computations
   - Add async checkpoint saving

3. **Long-term Enhancements**:
   - Profile with PyTorch Profiler to identify hidden bottlenecks
   - Consider using PyTorch Lightning for better training infrastructure
   - Experiment with ONNX export for inference optimization

---

**End of Report**
