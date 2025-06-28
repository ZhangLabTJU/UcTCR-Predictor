## **UcTCRPredictor**

The **Unconventional-TCR Predictor** is a lightweight Python package that wraps our pre-trained **TCR-V-BERT** models for **human** and **mouse** repertoires.

It embeds each TCR-β chain (CDR3 AA + V gene) and assigns unconventional T-cell probabilities:

- MAIT
- iNKT
- CD8αα IELs
- Conv (conventional CD4/CD8)

The package supports **command-line** and **pure-Python** workflows and can process millions of sequences in a single run.

---

### **License**

Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) (see [License file](https://github.com/ZhangLabTJU/UcTCR-Predictor/blob/main/LICENSE)). This software is not to be used for commercial purposes.

---

### **1 Installation**

```
# ❶ create / activate your env (optional)
conda create -n uctcr python=3.9 -y
conda activate uctcr

# ❷ clone & install
git clone https://github.com/ZhangLabTJU/UcTCR-Predictor.git
cd UcTCRPredictor
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
# Human model
uctcr --species human path/to/human_data.csv -o human_pred.tsv.gz

# Mouse model
uctcr --species mouse path/to/mouse_data.csv -o mouse_pred.tsv.gz
```

*Output* – a tsv.gz containing all original columns **plus** the four probability columns.

### **2.2 Python API**

```
from UcTCRPredictor.species.human import ucpredict_human   # or ucpredict_mouse

out = ucpredict_human(
    "path/to/human_data.csv",
    batch_size=2048,               # optional (default 1024)
    save_path="human_pred"         # optional, auto adds .tsv.gz
)
print(out.head())
```

---

> **Heads-up — model download on first run**
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
| **Vgene** | string | IMGT V gene short name (e.g. TCRBV05-01), reference V gene list is provided above. |
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
| iNKT | invariant NKT-cell probability |
| CD8αα IELs | intra-epithelial lymphocyte probability |

Probabilities in each row sum to **1.0**.

---

### **5 Model overview**

The **TCR-V-BERT** backbone jointly embeds **V-gene tokens** and **CDR3 amino-acid tokens**.

The frozen encoder feeds a shallow **2-layer feed-forward classifier**, yielding class probabilities that closely match expert-curated annotations (AUROC ≈ 0.96 on held-out datasets).

---

### **6 Citation**



---

Issues & PRs are welcome → [GitHub Issues](https://github.com/ZhangLabTJU/UcTCR-Predictor/issues)
