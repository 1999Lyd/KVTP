# KVTP
# Getting Started

## Installation and Setup
1. **Install the environment from the main directory**:
   ```bash
   conda create --name kvtp python==3.10
   conda activate kvtp
   pip install -e .
   ```

2. **Install the evaluation package**:
   ```bash
   cd lmms-eval
   pip install -e .
   ```

3. **Download and extract the data and checkpoints**:  
   Obtain all the necessary data and checkpoints from [this link](https://drive.google.com/drive/folders/16CMueWrw2HAJSWARRlitusxz-JhbYvR9?usp=sharing) and extract them into the main directory.

---

## Evaluation
1. **Run the inference script**:
   ```bash
   bash scripts/kvtp_infer.sh
   ```

2. **Switch the pruning method**:  
   To change the pruning method, update the `prune_method` field in `model_args` within the scripts. The available options are:
   - `prumerge`
   - `tome`
   - `random`
   - `None`
   To evaluate under hard selection, add args `if_hard=True` in model_args.
   
3. **Evaluate without a trained predictor**:  
   By removing the path to the predictor, the evaluation will be performed with the untuned Siglip model.

---

## Predictor Training
To train the predictor, run:
```bash
bash scripts/train_siglip.sh
```
```

## Citation
This repository was built on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT.git) and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
```