# Image Captioning with Transformers

A transformer-based image captioning model trained from scratch on COCO 2017 — InceptionV3 CNN encoder feeding a transformer encoder–decoder that generates natural-language captions for images, with beam search decoding and per-word attention visualization.

**[Live demo](https://huggingface.co/spaces/Akku0902/image-captioning-weights)** · **[Model weights](https://huggingface.co/Akku0902/image-captioning-weights)**

## What it does

Upload an image, and the model generates a caption word-by-word using beam search, while tracking which region of the image each word attended to. The deployed app visualizes this by pinning each generated word to its peak attention location on the image.

## Architecture

- **Encoder (visual):** InceptionV3 (ImageNet-pretrained, used as a frozen feature extractor) → reshaped feature map → 1 transformer encoder layer (single-head self-attention, embedding dim 512)
- **Decoder (language):** Transformer decoder with 8-head self-attention + 8-head cross-attention over encoder outputs, embedding dim 512, feed-forward width 512, dropout 0.1–0.5
- **Vocabulary:** 11,691 tokens, built via `TextVectorization` over COCO captions
- **Decoding:** Beam search (width 3, length penalty 0.7) at inference time

The captioning model itself — encoder and decoder — is trained entirely from scratch. Only the InceptionV3 backbone uses ImageNet-pretrained weights, purely as a fixed visual feature extractor.

## Results

Trained on 70K sampled COCO 2017 image-caption pairs (80/20 train/val split, ~56K/14K).

| Metric | Value |
|---|---|
| Best validation loss | 2.8233 (epoch 10) |
| Best validation accuracy (token-level) | 0.4504 |
| Training ran to | Epoch 13 (`EarlyStopping`, patience 3, restored epoch 10 weights) |
| **BLEU-4 (validation, beam search width 3)** | **0.0796** |

Training was resumed from a prior checkpoint rather than started from scratch, to work around Kaggle session time limits — optimizer: Adam, lr 1e-4, batch size 64.

## What failed, and what I learned

**A 10x BLEU inflation bug.** An early evaluation pass multiplied the final BLEU score by 10 before reporting it — a leftover scaling artifact that made the model look dramatically better than it was. Caught it by manually inspecting generated captions against the reported score and noticing the mismatch; the real, unscaled BLEU-4 (0.0796) is what's reported above and used in the deployed app.

**Token-level accuracy is a misleading metric here.** An early evaluation attempt used scikit-learn's `precision_score`/`confusion_matrix` over flattened token predictions and reported ~97.5% accuracy — which looked great but was almost entirely driven by the model correctly predicting padding tokens (mostly zeros in a sparse, long-tail vocabulary). Dropped this in favor of BLEU-4 over decoded, non-padded caption text, which actually reflects caption quality.

**Keras 3's stricter weight-file naming.** `model.save_weights()` now requires a `.weights.h5` suffix — calls using a plain `.h5` name fail outright. Hit this twice across different training runs before standardizing on the correct filename throughout.

**Kaggle session interruptions.** Long training runs (500+ seconds/epoch) risked disconnection mid-run. Mitigated by checkpointing weights every epoch (`ModelCheckpoint`, `save_best_only=False`) and warm-starting subsequent runs from the last saved checkpoint instead of retraining from zero.

## Tech stack

TensorFlow/Keras 3 · InceptionV3 · Streamlit (deployment) · Hugging Face Hub (model weight hosting) · trained on Kaggle (Tesla T4 GPU)

## Repo contents

- `app.py` — Streamlit inference app (loads weights from Hugging Face Hub, beam search + attention visualization)
- `ic-prject-coco-dataset.ipynb` — training notebook (data prep, model definition, training loop, evaluation)
- `vocab_coco.file` — pickled vocabulary used by the tokenizer (must match the vocabulary the deployed weights were trained with)

Model weights (`model.weights.h5`, ~465MB) are hosted on Hugging Face Hub rather than this repo, since GitHub rejects files over 100MB.
