# Multi-Class-IDS

## An observation that applies to both datasets
As an example, using CICIDS dataset. The large validation sets are murdering us.  We have 11 samples of class 14 and we're locking away more than half from training.  Might be smart to do away with the validation set, or to severely limit its size.

![image](https://github.com/user-attachments/assets/90a5b5ca-253f-49d5-a885-2a40fa7b7e87)

![image](https://github.com/user-attachments/assets/4a385277-cae7-461a-9e7b-c2b691e825ce)

![image](https://github.com/user-attachments/assets/294ea580-580a-4cf9-a48d-e5c48faca999)

![image](https://github.com/user-attachments/assets/1eb9086b-4519-416c-8abc-49656ab335aa)



## A simple instantiation of the quantum layer

```
class CosineQuantumLayer(nn.Module):
    """Fast surrogate for the Ry+CNOT ring producing ⟨Z_i⟩."""
    def __init__(self, n_qubits):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_qubits) * torch.pi)

    def forward(self, data):            # data : (B, n_qubits)
        return torch.cos(self.scale * data)   # (B, n_qubits)
class HybridSideBySideFastNN(nn.Module):
    def __init__(self, input_dim, num_classes, n_qubits=8):
        super().__init__()
        self.classical = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 16),        nn.ReLU()
        )
        self.q_compress = nn.Linear(16, n_qubits)     # 16 → 8
        self.q_layer    = CosineQuantumLayer(n_qubits)

        self.output = nn.Sequential(
            nn.Linear(n_qubits + 16, 16), nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):                          # x : (B, input_dim)
        x_emb  = self.classical(x)                 # (B, 16)
        q_in   = self.q_compress(x_emb)            # (B, 8)
        z_exp  = self.q_layer(q_in)                # (B, 8)

        features = torch.cat([x_emb, z_exp], dim=1)  # (B, 24)
        return self.output(features)
```


This instantiates this quantum circuit  
  
![image](https://github.com/user-attachments/assets/2da1c347-b594-45d0-8bef-a79b018b8d98)


This network looks like this

![image](https://github.com/user-attachments/assets/1565c940-4df9-4569-bb4f-e5f1a6a66836)


and it gives us these results.

Epoch 30/30
100%
 1934/1934 [00:21<00:00, 69.24it/s]
✅  New best model (val AUC 0.9989)
Val Acc: 0.9954 | Prec: 0.7708 | Rec: 0.6902 | F1: 0.7021
Test Acc: 0.9953 | Prec: 0.8404 | Rec: 0.6767 | F1: 0.7095
Test Classification Report:
|   class        |   precision |    recall |  f1-score |   support |
| --- | --- | --- | --- | --- |
|           0 |    0.9972 |   0.9971  |  0.9971  |  340611 |
|           1 |    0.9923 |   0.9959  |  0.9941  |   23694 |
|           2 |    0.9286 |   0.1193  |  0.2114  |     218 |
|           3 |    0.0000 |   0.0000  |  0.0000  |     128 |
|           4 |    0.0000 |   0.0000  |  0.0000  |       3 |
|           5 |    0.9949 |   0.9898  |  0.9923  |    1176 |
|           6 |    0.9623 |   0.9841  |  0.9731  |     881 |
|           7 |    0.9997 |   0.9990  |  0.9993  |   19244 |
|           8 |    0.9035 |   0.3705  |  0.5255  |     278 |
|           9 |    1.0000 |   0.2500  |  0.4000  |       4 |
|          10 |    0.9718 |   0.9852  |  0.9785  |     876 |
|          11 |    0.8888 |   0.9879  |  0.9357  |     825 |
|          12 |    0.9812 |   0.9914  |  0.9863  |   34607 |
|          13 |    0.9852 |   0.9798  |  0.9825  |    1635 |
|          14 |    1.0000 |   0.5000  |  0.6667  |       2 |
| | | | | |
|    accuracy  |           |          |  0.9953 |   424182 |
|   macro avg  |   0.8404  |  0.6767  |  0.7095 |   424182 |
| weighted avg |    0.9949 |   0.9953 |   0.9949 |   424182 |


## Latest Results from UNSW dataset

My batch sampler

```
class StratifiedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size = BATCH_SIZE):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.label_to_indices = defaultdict(list)

        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.class_probs = {
            label: len(indices) / len(labels)
            for label, indices in self.label_to_indices.items()
        }

    def __iter__(self):
        all_indices = []
        num_classes = len(self.label_to_indices)

        # Approximate how many of each class to sample per batch
        per_batch = {
            label: max(1, int(self.class_probs[label] * self.batch_size))
            for label in self.label_to_indices
        }

        num_batches = len(self)
        for _ in range(num_batches):
            batch = []
            for label, indices in self.label_to_indices.items():
                selected = np.random.choice(indices, per_batch[label], replace=True)
                batch.extend(selected)
            np.random.shuffle(batch)
            yield batch[:self.batch_size]  # trim excess if needed

    def __len__(self):
        return len(self.labels) // self.batch_size
```

#### What this does
It ensures the presence of at least some of each minority class in each batch.  We can tailor how balanced the sample is via batch size--smaller batches will be more balanced.

### Ran with 1024 batch size and my new batch sampler

✅ New best model saved at epoch 37 with val_auc = 0.9966
Epoch: 37, Loss: 0.0527
Val AUC: 0.9966 | Val F1: 0.4733
Test AUC: 0.9968 | Test F1: 0.4768
Sample-Weighted Accuracy: 0.9784
Category-Weighted Accuracy (Macro): 0.5510
Category-Weighted Accuracy (Weighted by Class Size): 0.9784

Per-Class Test Metrics:


| class | precision | recall | f1-score | support |  
| --- | --- | --- | --- | --- |
| 0 | 0.4000     | 0.0075   |   0.0147       |  535 |   
| 1 | 0.3684     | 0.0195    |  0.0370       |  359 |   
| 2 | 0.1503     | 0.8879     | 0.2571       |  107 |   
| 3 | 0.9931    |  0.9975     | 0.9953     | 443753 |   
| 4 | 0.3487    |  0.0556    |  0.0960      |  3271 |   
| 5 | 0.6009    |  0.8569     | 0.7064       | 8905 |   
| 6 | 0.6425    |  0.3863     | 0.4825      |  4849 |   
| 7 | 0.9937    |  0.9828     | 0.9882      | 43096 |   
| 8 | 0.8516 |     0.7320     | 0.7872     |   2798 |   
| 9 | 0.4769 |     0.6490     | 0.5498  |       302 |   
| 10 | 0.2500 | 0.4857     | 0.3301  |        35 |   
|      accuracy |           |          |   0.9784 |   508010 |
|    macro avg |    0.5524  |  0.5510  |  0.4768  |  508010 |
| weighted avg |    0.9764  |  0.9784  |  0.9757  |  508010 |

Memory allocated: 0.00 GB
Memory reserved: 0.00 GB



## ✅ Next Steps

- **Adaptive batch sizing**: Vary batch size across epochs to gradually transition from class-balanced learning to real-world distribution exposure.
- **Category weighting**: Use class-weighted loss functions (e.g., `weight=` in `CrossEntropyLoss`) to prioritize underrepresented classes.
- **Implement side-by-side experiments on cloud**: Evaluate the effect of different curriculum learning schedules, architectures, and loss functions on standardized metrics.
- **Streamline and modularize code**: Refactor training logic, batch samplers, dataset creation, and evaluation into reusable modules.
- **Save processed data into a Kaggle dataset**: Export the stratified, labeled, and scaled datasets for reproducibility and public benchmarks.
- **Need to update adaptive LR based on Loss**: 

---

## 🔍 Kill Chain-Based Grouping Recommendation

The current class labels can be grouped based on the **Cyber Kill Chain** to reflect real-world attacker stages:

| Kill Chain Phase       | Categories                                 |
|------------------------|--------------------------------------------|
| Reconnaissance         | `RECONNAISSANCE`, `ANALYSIS`               |
| Delivery & Probing     | `FUZZERS`, `GENERIC`                       |
| Exploitation           | `EXPLOITS`, `SHELLCODE`                    |
| Installation           | `BACKDOOR`, `BACKDOORS`, `WORMS`           |
| Impact                 | `DOS`                                      |
| Benign / No Threat     | `BENIGN`                                   |

- This grouping reflects attacker **intent and behavior** rather than statistical frequency.
- Group-based accuracy and confusion evaluation could highlight which *phases* of attacks are better understood by the model.
- These groups can also serve as a **superclass for hierarchical evaluation or modeling**.

---

## 🧠 Other Possible Improvements

### Curriculum & Sampling

- **Community of Experts system**:
  - A *Rare Attack Expert* model distinguishes rare attacks (positive) from benign + common attacks (negative).
  - Its output can act as a binary flag, an input feature, or a gating signal for a full classifier.
  - This model can also benefit from curriculum learning with enforced rare-class exposure.

- **Curriculum scheduling**:
  - Gradually relax enforced class balance.
  - Transition based on class-level confidence, entropy, or plateaued performance.

- **Category-aware sampling**:
  - Ensure each batch includes one or more examples from every class (if available).
  - Smaller batches early on promote meaningful exposure to rare classes.

- **Entropy-based routing**:
  - Use prediction uncertainty to route samples to fallback classifiers or expert branches.

- **Balanced sampling without replacement**:
  - Cycle through rare categories across epochs in a structured manner.

---

### Loss Functions

- **Cross Entropy Loss (baseline)**  
- **Class-weighted Cross Entropy**  
- **Focal Loss**: Emphasizes difficult or minority examples  
- **LDAM Loss**: Enforces larger margins for rare classes  
- **Label Smoothing**: Reduces overconfidence in dominant classes  
- **Contrastive Loss**: Useful for unsupervised or hybrid pretraining

---

### Optimizers & Scheduling

- **Adam / AdamW / RAdam**
- **SGD + Momentum**: More sensitive to batch quality, may benefit curriculum
- **Lookahead Optimizer**: Adds stability
- **Learning Rate Schedulers**:
  - `OneCycleLR`, `CosineAnnealing`, or plateau-based adjustments

---

### Modeling & Architectures

- **Mixture-of-Experts (MoE)**: Gated network architecture where subnetworks specialize in different classes or superclasses.
- **Rare vs Common Hybrid Classifier**: Binary classifier distinguishes “rare” vs “not-rare” then passes to full classifier.
- **Contrastive Pretraining + Supervised Fine-Tuning**

---

### Evaluation & Monitoring

- **Macro vs Weighted Precision/Recall/F1**
- **Per-Class AUC and Calibration**
- **GPU Utilization & Profiling** (`torch.profiler`, Kaggle’s resource monitor)
- **Memory Tracking**: `torch.cuda.memory_allocated()`, etc.

---

### Class Similarity & Semantically-Aware Evaluation

- **Problem with top-k metrics**:
  - (DOG or DUCK) ≠ (CHICKEN or TURKEY); proximity matters
- **Alternative strategies**:
  - Confusion matrix clustering
  - Manual taxonomy (e.g., propagation vs stealth vs denial)
  - Group-aware metrics (e.g., phase-level accuracy)
- **Future Work**:
  - Hierarchical prediction
  - Cost-sensitive loss tied to group similarity
  - Logit-space analysis for class neighborhoods
