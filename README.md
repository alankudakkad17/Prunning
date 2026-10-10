# Model Pruning with PyTorch

A hands-on walkthrough of three pruning strategies from `torch.nn.utils.prune`, finishing with a practical global-pruning comparison on a pre-trained ResNet18.

## What's in the notebook

### Part 1 — Pruning mechanics on a dummy model
All three techniques are first demonstrated on a tiny `SimpleModel` (one `Conv2d` layer `conv1`, one `Linear` layer `fc1`), re-initialized fresh before each section so every technique starts from the same unpruned state.

1. **Unstructured pruning** (`prune.l1_unstructured`) — removes the 30% of individual weights in `conv1` with the smallest L1 magnitude, scattered anywhere in the weight matrix. Architecture stays unchanged; needs hardware/software support for sparsity to pay off in speed.
2. **Structured pruning** (`prune.ln_structured`) — removes 50% of entire output neurons (rows) in `fc1`, ranked by L2 norm. Because whole neurons disappear, this can speed up inference without special hardware.
3. **Global pruning** (`prune.global_unstructured`) — removes the weakest 20% of weights across `conv1` and `fc1` *combined*, so each layer ends up with a different, unequal sparsity depending on which weights were globally least important.

Each section follows the same pattern: inspect state → apply pruning → verify (`weight_mask` / `weight_orig` appear, sparsity stats, zeroed values) → `prune.remove()` to make it permanent → verify it's back to a clean, standard module.

### Part 2 — Practical application: pruning a pre-trained ResNet18
1. Load the [Fruit and Vegetable Disease (Healthy vs Rotten)](https://www.kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten) subset (28 classes, 200 images each) via `helper_utils.get_dataloaders`.
2. Train an **unpruned baseline**: ResNet18 (ImageNet weights) with its final layer replaced for 28 classes, fine-tuned for 2 epochs, all layers trainable.
3. Train a **pruned model**: a second ResNet18 gets 50% global unstructured pruning applied to every `Conv2d`/`Linear` layer *except* the final classifier, then is fine-tuned for 2 epochs *before* `prune.remove()` is called — so the optimizer can adapt the surviving weights around the permanently-zeroed ones.
4. `helper_utils.comparison_report()` prints a final side-by-side: total vs. non-zero parameters, saved file size, and accuracy/precision/recall/F1 for both models.

**Key takeaway the notebook ends on:** the pruned model keeps ~50% fewer *effective* (non-zero) parameters with comparable accuracy, but the saved `.pth` file is the *same size* as the unpruned one — `torch.save` stores the zeros densely. Actually shrinking the file requires additional steps (e.g. sparse storage formats) or a different compression technique like quantization.

## Files
- `C3_M4_Lab_3_pruning_annotated.ipynb` — the lab notebook with a "What I did" note added before every code cell
- `helper_utils.py` — must sit in the same directory; supplies:
  - `show_weights` — prints a sample of a layer's weights for visual inspection
  - `get_dataloaders`, `load_resnet18`, `replace_final_layer`, `training_loop` — data/model/training pipeline
  - `save_unpruned_model_and_metrics`, `save_pruned_model_and_metrics`, `comparison_report` — persistence and final comparison

## Requirements
- Python with `torch`, `torchvision`, `torchmetrics`, `tqdm`, `numpy`
- `./fruit_and_vegetable_subset/` dataset directory (ImageFolder structure, 28 class subfolders)
- `./pretrained_resnet18_weights/resnet18-f37072fd.pth` — local ImageNet-pretrained ResNet18 weights
- CPU or CUDA GPU (auto-detected)

## Outputs produced when run
- `unpruned_model_state_dict.pth`, `unpruned_metrics.pkl`
- `pruned_model_permanent_state_dict.pth`, `pruned_metrics.pkl`

## How to use
Run top to bottom. Each code cell has a short **"What I did"** note directly above it; use those as a running commentary of the lab. This notebook sets up the segue into **quantization** (a separate lab), which actually reduces model file size by lowering numeric precision — something pruning alone doesn't do.
