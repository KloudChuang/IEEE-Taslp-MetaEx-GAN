# MetaEx-GAN: Meta Exploration to Improve Natural Language Generation via Generative Adversarial Networks

Official implementation of the paper **"MetaEx-GAN: Meta Exploration to Improve Natural Language Generation via Generative Adversarial Networks"**, submitted to *IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)*.

**Authors:** Yun-Yen Chuang, Hung-Min Hsu, Kevin Lin, Ray-I Chang, Hung-Yi Lee
*(National Taiwan University · University of Washington · Microsoft Azure AI)*

---

## Overview

Generative Adversarial Networks (GANs) applied to text — so-called **Language GANs** — are typically trained with reinforcement-learning methods such as policy gradients (e.g., SeqGAN's Monte-Carlo roll-out). A central limitation is that the **same generator is responsible for both sampling (exploration) and learning (exploitation)**. Because the exploration strategy is identical to the exploitation strategy, these models cannot explore unexplored regions of the sequence space effectively, and they **cannot improve sampling quality and diversity at the same time**. Common workarounds (e.g., enlarging the roll-out batch size, as in ScratchGAN) increase the number of generated sequences and are computationally unfair to compare against.

**MetaEx-GAN** addresses this by introducing a separate **explorer** trained via **Meta Exploration (MetaEx)** in a *teacher–student* framework:

- The **explorer** (teacher, `Gψ`) meta-learns *exploration strategies* and is dedicated to **sampling** the data used to train the generator.
- The **generator** (student, `Gθ`) is dedicated to **learning** and reports back its *learning effectiveness*.
- The explorer is updated by a **meta-reward** defined as the improvement in the generator's reward after being trained on the explorer's samples.

This decoupling of exploration from exploitation yields **better sampling efficiency without increasing the number of generated sequences**, and is **model-agnostic** — it can be layered on top of existing Language GANs (SeqGAN, RankGAN, LeakGAN, RelGAN, MetaCotGAN, ScratchGAN) and large-scale pre-trained models (GPT-2).

### Key contributions

1. **MetaEx-GAN** — a meta-learning–based Language GAN that decouples sampling (explorer) from learning (generator) to improve NLG.
2. A practical recipe for applying a **MetaEx-trained explorer** to Language GANs to reach **state-of-the-art** performance.
3. **Backtracking** — an essential training trick for stabilising the meta-learning of the generator (see *Method*).
4. **Generality** — MetaEx-GAN can be combined with other Language GAN architectures *and* with GPT-2 (as generator, explorer, or both).
5. **Visualization** of the explorer's behaviour, showing how it samples more diverse trajectories around the real-data distribution.

---

## Method

MetaEx-GAN formulates NLG as a sequential decision process. For a sequence `Y_{1:T}`, the state `s` at step `t` is the prefix `(y_1, …, y_{t-1})` and the action `a` is the next token `y_t`.

### Components

| Symbol | Role | Notes |
|--------|------|-------|
| `Gθ`   | **Generator** (student) | Learns; same architecture/settings as the SeqGAN generator. |
| `Gψ`   | **Explorer** (teacher)  | Meta-learns exploration; performs the roll-out sampling. Initialised from the pre-trained generator. |
| `Dϕ`   | **Discriminator**       | Distinguishes real vs. generated sequences; provides the reward. |

### Training loop (Algorithm 1)

1. **Pre-training.** Pre-train `Gθ` with MLE; pre-train `Dϕ` with the standard GAN discriminator loss (Eq. 1).
2. **Exploration epochs (`e = 1 … E`).** For each exploration epoch:
   - The **explorer** rolls out to produce samples and the **Q-function** `Q^{Gψ}_{Dϕ}` is computed (Eq. 7).
   - The generator gradient `∇J(θ)_e` is computed from the explorer's samples (Eq. 8) and a *temporary* update `θ_e ← θ + ∇J(θ)_e` is applied (Eq. 5).
   - The **meta-reward** is estimated as the improvement in the generator's reward:
     `R_{Gψ} = R(Y^{Gθ′}) − R(Y^{Gθ})`  (Eq. 9), where `R(Y) = average(Dϕ(Y))`.
   - The explorer gradient `∇J(ψ)_e` is accumulated (Eq. 10).
   - **Backtracking:** recover `θ_e → θ` so every exploration epoch evaluates a *different* batch against the *same* generator weights.
3. **Explorer update.** `ψ′ ← ψ + Σ_e ∇J(ψ)_e` (Eq. 11).
4. **Generator update.** Re-sample with the updated explorer `Gψ′` and apply the real generator update `θ′ ← θ + ∇J(θ)` (Eq. 5/8).
5. **Discriminator update** (Eq. 1).
6. Repeat until convergence.

### Training tricks

- **Exploration epochs.** Collect a *diverse set of meta-rewards* (different seeds) before updating the explorer, enabling better training of the deterministic generator policy.
- **Backtracking.** Restore the generator to its weights at the start of the exploration epochs so the explorer meta-learns the learning-effectiveness of the *same* generator across different sampled batches. This is shown to be **essential** for performance.
- **Constant compute.** To avoid enlarging the number of generated sequences, the batch size per exploration epoch is set to `B/E`. With `E` exploration epochs the total number of generated sequences in a training epoch stays at `B`, giving a **fair comparison** to other Language GANs.

---

## Repository structure

| File | Description |
|------|-------------|
| `MetaEx-GAN.py` | Main training script (oracle, generator, explorer, discriminator, MetaEx training loop). Implemented in **TensorFlow 2.0**. |
| `generator.py` | Generator / explorer network definition. |
| `discriminator.py` | Discriminator network definition. |
| `utils.py` | Helper utilities (data handling, metrics, etc.). |
| `Function description of MetaEx-GAN.ipynb` | Notebook walking through the code in detail. |
| `training data.zip` | Training data. |
| `test data.zip` | Test data. |
| `experiment result.zip` | Saved experiment results. |
| `MetaEx-GANvsScratchGAN.png` | Comparison figure. |

---

## Requirements

- Python 3
- **TensorFlow 2.0** (this work is implemented with TF 2.0 / `tf.keras`)
- NumPy, scikit-learn, tqdm
- A CUDA-capable GPU is recommended (the script enables GPU memory growth automatically).

```bash
pip install "tensorflow>=2.0" numpy scikit-learn tqdm
```

---

## Usage

```bash
# Unzip the provided datasets
unzip "training data.zip"
unzip "test data.zip"

# Run training / evaluation
python MetaEx-GAN.py
```

The notebook **`Function description of MetaEx-GAN.ipynb`** provides a step-by-step explanation of each component.

Default hyper-parameters (see the top of `MetaEx-GAN.py`):

| Hyper-parameter | Value |
|-----------------|-------|
| `max_length` (sequence length) | 20 |
| `num_words` (vocabulary) | 5000 |
| `batch_size` | 64 |
| MLE pre-training epochs | 150 (paper) |
| Discriminator pre-training epochs | 150 (paper) |
| Adversarial generator/discriminator alternation | 15× |
| Total training epochs | 1000 |
| Roll-out samples `N` | 64 |

> In the paper, all Language-GAN results are averaged over **6 runs** (random seeds). ScratchGAN uses `N = 512` and no MLE pre-training.

---

## Datasets

- **Synthetic data** (oracle-LSTM, following SeqGAN): 5,000 words, sentence length 20, 10,000 oracle-generated sentences. Evaluated with **NLL_oracle**.
- **COCO Image Captions:** 4,682 unique words, max length 37, 10,000 train / 10,000 test sentences.
- **EMNLP2017 WMT News:** 5,255 unique words, max length 51, 270,000 train / 10,000 test sentences.

---

## Evaluation metrics

- **NLL_oracle** (↓) — sampling quality on synthetic data.
- **BLEU-5** (↑) — sampling quality on real data.
- **Self-BLEU-5** (↓) — sampling diversity. (Quality–diversity curves are produced via softmax-temperature sweeps.)
- **FED — Fréchet Embedding Distance** (↓) — jointly measures quality + diversity using Universal Sentence embeddings.
- **Human evaluation** (↑) — 100 native English speakers grade 40 generated COCO captions (1–10).

---

## Results (selected)

**Synthetic data — NLL_oracle (↓), roll-out N = 64:**

| Model | NLL_oracle |
|-------|-----------:|
| SeqGAN | 8.736 |
| LeakGAN | 7.038 |
| RelGAN | 6.680 |
| MetaCotGAN | 7.690 |
| **MetaEx-GAN** | **6.321** |

**Model-agnostic — FED (↓) on EMNLP2017 News (each base method + MetaEx-GAN):**

| Base method | Base FED | + MetaEx-GAN |
|-------------|---------:|-------------:|
| SeqGAN | 0.1422 | **0.0180** |
| RankGAN | 0.1431 | **0.0195** |
| LeakGAN | 0.0691 | **0.0189** |
| RelGAN | 0.0408 | **0.0177** |
| MetaCotGAN | 0.0375 | **0.0179** |
| ScratchGAN | 0.0390 | **0.0172** |

**With GPT-2 — FED (↓) on EMNLP2017 News:**

| Model | FED |
|-------|----:|
| GPT-2 + MLE | 0.0202 |
| TextGAIL (GPT-2 / RoBERTa) | 0.0185 |
| **MetaEx-GAN + GPT-2** | **0.0145** |

**Sampling efficiency — EMNLP2017 News (averaged over 6 runs):** MetaEx-GAN reaches its best FED (0.0180) in only **95 ± 18 epochs**, versus 115–952 epochs for the baselines, with the lowest variance across metrics.

**Human evaluation — COCO (↑):** MetaEx-GAN scores **6.12** and MetaEx-GAN + GPT-2 **6.32**, the highest among Language GANs (human-written reference: 7.42).

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{chuang2023metaexgan,
  title   = {MetaEx-GAN: Meta Exploration to Improve Natural Language Generation
             via Generative Adversarial Networks},
  author  = {Chuang, Yun-Yen and Hsu, Hung-Min and Lin, Kevin and
             Chang, Ray-I and Lee, Hung-Yi},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year    = {2023}
}
```

---

## Acknowledgements

The experimental setup follows the [Texygen](https://github.com/geek-ai/Texygen) benchmarking platform and prior Language-GAN works (SeqGAN, LeakGAN, RelGAN, MetaCotGAN, ScratchGAN, TextGAIL). The GPT-2 experiments follow the TextGAIL setting (GPT-2-base generator, RoBERTa-base discriminator).
