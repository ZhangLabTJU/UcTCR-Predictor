## **UcTCR-Predictor**

UcTCR-Predictor is a lightweight and scalable Python package designed for rapid identification of unconventional αβ T cells from bulk TCRβ repertoire data. The core of the framework is TCR-V-BERT, a transformer-based model pretrained on large-scale unlabeled TCRβ sequences using a self-supervised masked language modeling strategy. This pretraining captures generalizable sequence features, which are then fine-tuned for multi-class classification of unconventional T cell subsets, including MAIT and iNKT cells. The two-stage design—unsupervised pretraining followed by supervised transfer—enables robust generalization across diverse datasets, tissue sources, and immune contexts. UcTCR-Predictor supports both  **human** and **mouse** repertoires, with species-specific models optimized for each organism. 

It embeds each TCR-β chain (CDR3 AA + V gene) and assigns unconventional T-cell probabilities:

- MAIT cells
- iNKT cells
- Conv (conventional CD4/CD8) T cells

The package supports **command-line** and **pure-Python** workflows and can process millions of sequences in a single run.

---

### **License**

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) (see [License file](https://github.com/ZhangLabTJU/UcTCR-Predictor/blob/main/LICENSE)). This software is not to be used for commercial purposes.

---

### **1 Installation**

```
# 1. create / activate your env (optional)
conda create -n uctcr python=3.9 -y
conda activate uctcr

# 2. clone & install
git clone https://github.com/ZhangLabTJU/UcTCR-Predictor.git
cd UcTCR-Predictor
pip install -e .        # dev-mode; use  "pip install ."  for prod
```

> The install pulls these core deps:
> 
> 
> **torch**
> 
> **transformers**
> 
> **pandas**
> 
> **numpy**
> 
> huggingface-hub
> 

---

### **2 Quick usage**

### **2.1 CLI (recommended for large files)**

```
# command: UcTCRPredictor or uctcrp
# Human model
# uctcrp (or UcTCRPredictor) --species human <input path> -o <output path>
uctcrp --species human tests/test_data_human.csv -o human_pred

# Mouse model
# uctcrp (or UcTCRPredictor) --species mouse <input path> -o <output path>
uctcrp --species mouse tests/test_data_mouse.csv -o mouse_pred
```

*Output* – a tsv.gz containing all original columns **plus** the four probability columns.

### **2.2 Python API**

```
from UcTCRPredictor.species.human import ucpredict_human
# or mouse:
# from UcTCRPredictor.species.mouse import ucpredict_mouse

out = ucpredict_human(
    "tests/test_data_human.csv",   # input path
    batch_size=2048,               # optional (default 1024)
    save_path="human_pred"         # optional, auto adds .tsv.gz
)
print(out.head())
```

---

:warning: **Heads-up — model download on first run**
> 
> 
> The very first time you call `uctcr` (or `ucpredict_human / ucpredict_mouse`) the package will automatically fetch the frozen TCR-V-BERT backbone **from Hugging Face**:
> 
> - `human` model ≈ 345 MB
> - `mouse` model ≈ 345 MB
> 
> The weights are cached under
> 
> `UcTCRPredictor/species/<species>/models/` and reused for all subsequent runs, so the download happens **only once**.
> 
> Make sure your machine can reach `huggingface.co` (HTTP 443) during that first execution.
> 

### **3 Input-file format**

| **column** | **type** | **description** |
| --- | --- | --- |
| **Vgene** | string | The input V gene is expected to follow the standardized naming convention (e.g., TCRBV05-01). Valid V gene names are listed below for reference: [Human V genes](https://github.com/ZhangLabTJU/UcTCR-Predictor/blob/main/Reference_Human_V_Gene.csv), [Mouse V genes](https://github.com/ZhangLabTJU/UcTCR-Predictor/blob/main/Reference_Mouse_V_Gene.csv). |
| **cdr3aa** | string | For accurate analysis, the CDR3 amino-acid sequence is expected to start with “C” and end with “F”. |

**Minimum example**

```
Vgene,cdr3aa
TCRBV05-01,CASSPTVFKGWRTEAFF
TCRBV14-01,CASSPWGKTQYF
```

*Lines containing invalid V genes or CDR3s are automatically filtered out.*

---

### **4 Output columns**

| **new column** | **description** |
| --- | --- |
| Conv | conventional αβ-T-cell probability |
| MAIT | mucosal-associated invariant T-cell prob. |
| iNKT | invariant nature killer T-cell probability |

Probabilities in each row sum to **1.0**.

---

### **5 Citation**



---

Issues & PRs are welcome → [GitHub Issues](https://github.com/ZhangLabTJU/UcTCR-Predictor/issues)
