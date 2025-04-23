# Multi-Class-IDS

## Latest Results from UNSW dataset

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
