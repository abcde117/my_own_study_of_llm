
---

# Lightweight RL-style Fine-tuning for Language Models in Resource-Constrained Settings

This project presents a custom, lightweight implementation of RL-style fine-tuning for language models, built entirely from scratch with **minimal computing resources and no access to human feedback datasets or APIs**.

The goal was not to outperform existing RLHF pipelines, but to **explore how much of reinforcement learning-inspired tuning can be replicated under extreme constraints**. The resulting framework includes custom generation logic, an approximate reward modeling system, and a PPO-style training loop — all running in a single notebook file.

> **Disclaimer**: The actual output quality is **not competitive with real RLHF pipelines**. This project was a **creative and educational effort** rather than a production-level system.

---

## Key Contributions and Novel Techniques

### 1. Custom Token-by-Token Generation with PCA Compression

Instead of relying on `transformers.generate()`, this project includes a manual token-by-token sampling loop. At each generation step, it applies `torch.pca_lowrank()` on the model’s logits and retains only the **first principal component**:

```python
u, s, vt = torch.pca_lowrank(logits)
logits = vt.transpose(1, 2)[:, 0, :]
```

This was done to **reduce the dimensionality and computation cost** of the softmax sampling step, especially useful when running on GPUs with limited memory.

> **Motivation**: Efficient inference under memory pressure. PCA reduces token logits size at the cost of fidelity.

---

### 2. Reward Model Based on Sentence Embedding Similarity

No human preference data was available, so the reward model is based on **semantic similarity**:

- Outputs from the language model are compared to ground-truth targets.
- Sentence embeddings are obtained using `sentence-transformers/all-MiniLM-L6-v2`.
- Cosine similarity scores are scaled and used as reward signals.
- A two-layer MLP is trained to imitate these similarity scores.

```python
score = cosine_similarity(embedding_output, embedding_label) * 10
```

This reward function is inherently **noisy and indirect**, but enabled fully unsupervised training without curated labels.

> **Innovation**: Approximating human feedback with embedding similarity as a proxy for semantic alignment.

---

### 3. Custom PPO + Advantage Actor-Critic Implementation

The project includes a hand-built PPO loop with the following components:

- Generalized Advantage Estimation (GAE)
- Entropy regularization
- KL penalty (between current and previous policy)
- Value clipping
- LoRA-based fine-tuning to minimize memory usage

```python
surr1 = ratios * (adv + delta)
surr2 = ratios.clamp(1-eps_clip, 1+eps_clip) * (adv + delta)
loss = -torch.min(surr1, surr2) - beta_entropy * entropy
```

All gradients are propagated and updated manually. There is **no external PPO or RL framework** used here.

> **Purpose**: Learnable fine-tuning process without relying on pre-existing RL tooling or API servers.

---

### 4. (Experimental) Use of FFT for Positional Encoding / Representation Mixing

Earlier in the project (not shown in this README), FFT was used to replace traditional positional encoding, inspired by the DFT mechanism. This was an experimental design for learning representations in a frequency domain rather than a spatial one.

Contrary to initial assumptions, the FFT-based positional encoding did not lead to repetitive outputs. In fact, compared to standard positional encodings (as in LLaMA), which occasionally collapsed into degenerate patterns such as "nine nine nine ...", the FFT-based model showed greater robustness. Notably, it was able to generate novel tokens not present in the original vocabulary — a behavior suggesting interpolation or synthesis in token space.

This indicates the potential of frequency-domain modeling for:

- Reducing repetition in autoregressive generation,
- Enabling creative synthesis beyond training data,
- Supporting better generalization in low-resource or compressed settings.

**Takeaway:** FFT-based positional modeling may offer advantages in generative diversity and compact expressiveness — even for pure text tasks — and opens the door to novel use cases such as encryption or semantic token remapping.


> **Takeaway**: FFT-based positional modeling might better suit multimodal or structured data than pure text generation.

---

## Limitations

The quality of generated text is limited and often repetitive or semantically weak.

The reward model is a crude proxy and doesn’t reflect real human preferences.

PCA projection causes loss of precision during generation.

Sentence-level rewards do not align well with token-level training signals.

FFT-based position encoding (if used) performs worse than RoPE or learned embeddings in terms of stability and control in large-scale generation.

This project is a proof of concept on what can be done with only a model, a tokenizer, and open-source tools — not a production-ready system.

---

## Structure

- `model`: Backbone Transformer + LoRA fine-tuning  
- `reward_model`: MLP trained on sentence similarity scores  
- `generate()`: Custom sampling loop with PCA compression  
- `ppo_loop`: Manual PPO with actor-critic and GAE  
- `gradio_demo`: Text input/output interface (optional)

---

# リソース制限下での軽量RL風言語モデル微調整

このプロジェクトは、**限られた計算資源**と**人間のフィードバックを使わない環境**でも可能な、RL（強化学習）スタイルの言語モデル微調整を、自作で実装した試みです。

目的は、実際のRLHF（人間のフィードバックによる強化学習）の再現ではなく、**制約された条件下でどこまでそれらしい手法が再現できるかの検証**でした。  
ノートブック1本で構成されており、カスタム生成ロジック、簡易リワードモデル、PPO風トレーニングループなどを実装しています。

> **注意**：生成品質はあまり高くなく、本格的なRLHFと比較できるものではありません。これは**教育的・創造的実験**として取り組んだものです。

---

## 主な工夫・独自実装ポイント

### 1. PCA圧縮を用いたトークン生成ループの自作

HuggingFaceの `.generate()` を使用せず、**1トークンずつ逐次生成**を実装しました。  
その際、各ステップのlogits（出力分布）に対し `torch.pca_lowrank()` を適用し、**第1主成分のみを利用**してsoftmaxを行います：

```python
u, s, vt = torch.pca_lowrank(logits)
logits = vt.transpose(1, 2)[:, 0, :]
```

> **動機**：GPUメモリが限られていたため、logitsの次元圧縮による計算量の削減を目的としました。

---

### 2. 文埋め込み類似度を用いた擬似報酬モデル

人間フィードバックデータが使えなかったため、**埋め込み間の意味的類似度**をスコアとするリワードモデルを構築：

- モデル生成文と正解文をSentenceTransformerでエンコード
- コサイン類似度をスコア化し、MLPに教師信号として与える
- 2層のLinear層のみで軽量実装

```python
score = cosine_similarity(embedding_output, embedding_label) * 10
```

> **特徴**：ラベル不要で弱教師的に報酬を生成する簡易RLHF風学習が可能になります。

---

### 3. PPO風のActor-Criticを自前実装

以下の構成でPPOを模倣しています：

- GAE（Generalized Advantage Estimation）
- エントロピー正則化
- KLダイバージェンス罰則
- Value Clipping
- LoRAによるメモリ効率化

```python
surr1 = ratios * (adv + delta)
surr2 = ratios.clamp(1-eps_clip, 1+eps_clip) * (adv + delta)
loss = -torch.min(surr1, surr2) - beta_entropy * entropy
```

> **注目点**：すべて手書きのPPO風アルゴリズムで、既存ライブラリに依存せずに完結しています。

---

### 4. 実験的なFFTベースの位置エンコーディング

プロジェクトの初期には、LLaMAやBERTに触発され、従来の位置エンコーディングを**FFT（高速フーリエ変換）で置き換える試み**も行いました。

当初の予想とは異なり、FFTベースの位置エンコーディングは繰り返し的な出力に陥らず、むしろ標準的な位置埋め込み（例：LLaMA）のように「nine nine nine...」といった退化したパターンを生み出すことが少なく、より頑健な動作を見せました。さらに、**語彙に存在しないトークンを合成・生成する挙動**も観察され、トークンスペース上での補間的・合成的な能力が示唆されます。

このことから、周波数領域に基づくモデリングには以下のような可能性があると考えられます：

- 自回帰生成における繰り返しの抑制
- 学習データ外の創発的表現の合成
- 圧縮・低資源状況下での汎化性能向上

**まとめ：** FFTベースの位置表現は、自然言語でも多様性やコンパクトな表現力に利点を持ち、暗号化や意味的トークン変換といった新たな応用への道も拓く可能性があります。


---

## 限界・注意点

- 出力文の品質は不十分で、意味的に弱く反復的な傾向があります  
- 報酬関数はあくまで近似であり、トークン単位の精度が高くありません  
- PCAによるlogits圧縮は精度低下を引き起こします  
- 文単位のスコアはトークン単位の訓練信号と整合しません  
- FFTベースの位置埋め込みは、RoPEや学習型埋め込みと比べて、大規模生成における安定性や制御性に劣る傾向があります  

このプロジェクトは、モデル・トークナイザ・オープンソースツールだけで何ができるかを示す**概念実証**であり、本番運用を想定したものではありません。


---

## 構成内容

- `model`: TransformerベースのLoRA適用モデル  
- `reward_model`: 文類似度に基づく軽量報酬モデル  
- `generate()`: PCAを用いたトークン逐次生成関数  
- `ppo_loop`: GAEとValue Clippingを含むActor-Criticループ  
- `gradio_demo`: UIでのインタラクティブテスト（任意）

---
