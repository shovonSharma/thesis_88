# thesis_88

DMoEBERT, a Systematic Study of Sparse Encoder Models Trained from Scratch under Resource Constraints, is my thesis work. I was inspired by META's [Dynamic-Tanh](https://arxiv.org/abs/2503.10622) to overcome normalization layer's computational overhead and Sparsely activated mixture of experts from GOOGLE's [Switch Transformers](https://arxiv.org/abs/2101.03961) for efficient feed forward network. I trained the complete pipeline from scratch on downstream tasks such as natural language inference, sentiment analysis, paraphrase detection, and topic classification.

For ablation studies, I compared my model against finetuned BERT-base(110M) and RoBERTA-base(125M) as well as custom configuration of BERT to match my models' parameters. For finetuning, i used hugging face framework PEFT/LoRA. DMoEBERT outperform those models in efficiency and shows competitive accuracy.

#### Rad the detailed report of my thesis [here](https://github.com/shovonSharma/thesis_88/blob/main/Report%20on%20DMoEBERT.pdf).

## ARCHITECTURE
![DMoEBERT Architecture](https://github.com/shovonSharma/thesis_88/blob/main/DMoEBERT.jpg)

### Model configuration of DMoEBERT (DM) variants
```
| Configuration   | DMoEBERT-23M   | DMoEBERT-45M | DMoEBERT-97M   |
|-----------------|----------------|--------------|----------------|
| embed_dim       |      256       |      384     |      512       |    
| Heads           |       4        |       6      |       8        |    
| Experts         |       6        |       6      |       6        |
| Layers          |       8        |       8      |      11        |
| capacity factor |     1.15       |     1.15     |     1.15       |
```

## DOWNSTREAM TASKS 

#### 🟩 Natural Language Inference - SNLI Example (Stanford Natural Language Inference)

Premise : A man inspects the uniform of a figure in some East Asian country.

Hypothesis : The man is sleeping.

Label : Contradiction

Explanation : The premise describes a man actively inspecting something — contradicts “sleeping.”

#### 🟦 Natural Language Inference - MNLI Example (Multi-Genre Natural Language Inference)

Premise (from "Government" genre) : The prime minister said the country will increase investment in renewable energy sources.

Hypothesis : The country plans to spend more money on wind and solar power.

Label : Entailment

Explanation : Investing more in renewable energy entails spending more on wind and solar power (common renewable types).

#### 🟥 Natural Language Inference - ANLI Example (Adversarial NLI)

Premise : The teacher warned the students that if they didn’t stop talking, they’d miss their recess.

Hypothesis : The students stayed quiet and went out for recess.

Label : Contradiction

Explanation : The premise implies they were talking and being warned — contradicting that they stayed quiet and got recess.

#### 🟥 Paraphrase Detection - QQP Example (Quora Question Pairs)

Question 1: How can I improve my English speaking skills?

Question 2: What are the best ways to get better at speaking English?

Label: Duplicate

Explanation: Both questions ask for methods to improve spoken English, so they have the same meaning.

#### 🟥 Sentiment Analysis - SST-2 Example (Stanford Sentiment Treebank v2)

Sentence: The movie had great acting but a painfully slow storyline.

Label: Negative

Explanation: Despite praising the acting, the sentence expresses an overall negative opinion due to the slow storyline

#### 🟥 topic classification - AG-NEWS Example 

Text: NASA announced a new mission to explore the surface of Mars using advanced robotics.

Label: Sci/Tech

Explanation: The text discusses a space mission and advanced technology, which falls under the Science/Technology category.



## Performance comparison and ablation study on SNLI and ANLI test sets. Acc.=Accuracy, F1-M=F1-macro
```
----------------------------------------------------------------------------------------------------------------------
| Model                          | SNLI Acc. | SNLI F1-M | ANLI-R2 Acc. | ANLI-R2 F1-M | ANLI-R3 Acc. | ANLI-R3 F1-M |
|--------------------------------|-----------|-----------|--------------|--------------|--------------|--------------|
| BERT-base (LoRA)               | 87.4      | 87.3      | 28.7         | 28.1         | 31.2         | 30.9         |
| RoBERTa-base (LoRA)            | 88.8      | 88.8      | 32.8         | 21.9         | 32.7         | 21.1         |
| BERT-23M                       | 80.0      | 80.0      | 29.6         | 29.1         | 27.9         | 27.8         |
| BERT-45M                       | 80.9      | 80.9      | 27.7         | 27.2         | 29.0         | 28.5         |
| DMoEBERT-23M                   | 82.7      | 82.6      | 27.7         | 27.6         | 30.7         | 30.6         |
| DMoEBERT-45M                   | 83.7      | 83.7      | 28.8         | 28.7         | 28.4         | 28.4         |
| DMoEBERT-97M                   | 84.0      | 84.0      | 28.7         | 28.6         | 30.8         | 30.5         |
| *Ablation Study*               |           |           |              |              |              |              |
| DMoEBERT-23M w/o LayerNorm     | 78.1      | 78.0      | 28.3         | 27.9         | 29.7         | 29.4         |
| DMoEBERT-45M w/o LayerNorm     | 79.6      | 79.4      | 28.5         | 28.1         | 28.8         | 28.6         |
| DMoEBERT-23M (+ANLI)           | 83.0      | 82.9      | 37.2         | 36.9         | 37.6         | 37.1         |
| DMoEBERT-45M (+ANLI)           | 83.6      | 83.6      | 40.3         | 40.1         | 38.7         | 38.4         |
| DMoEBERT-23M (DyT → LayerNorm) | 83.2      | 83.1      | 27.5         | 27.5         | 29.4         | 29.4         |
| DMoEBERT-45M (DyT → LayerNorm) | 82.0      | 82.0      | 28.7         | 28.5         | 28.5         | 28.5         |
| DMoEBERT-23M (MoE → FFWD)      | 82.9      | 82.8      | 27.2         | 27.1         | 30.7         | 30.6         |
| DMoEBERT-45M (MoE → FFWD)      | 80.4      | 80.4      | 27.9         | 27.5         | 28.8         | 28.7         |
----------------------------------------------------------------------------------------------------------------------
```

## Performance on AG News, QQP, and SST-2. Acc.=Accuracy, F1-M=F1-macro
```
| Model           | AG News Acc.| AG News F1-M| QQP Acc.* | QQP F1-M* | SST-2 Acc.* | SST-2 F1-M* |
|-----------------|-------------|-------------|-----------|-----------|-------------|-------------|
| BERT-23M        | 91.7        | 91.7        | 81.5      | 80.3      | 81.1        | 81.1        |
| DMoEBERT-23M    | 91.6        | 91.6        | 83.9      | 83.1      | 81.2        | 81.2        |

\* Validation set results reported for GLUE tasks (QQP, SST-2) as test labels are not publicly available.
```


## Per-class performance

#### Per-class performance on SNLI test set. Ent. = Entailment, Neu. = Neutral, Con. = Contradiction.
```
-------------------------------------------------------------------------------------------------------------------------------------------
| Model              | Precision Ent.| Precision Neu.| Precision Con.| Recall Ent.| Recall Neu.| Recall Con.| F1 Ent. | F1 Neu. | F1 Con. |
|--------------------|---------------|---------------|---------------|------------|------------|------------|---------|---------|---------|
| BERT-23M           | 0.84          | 0.74          | 0.82          | 0.84       | 0.75       | 0.80       | 0.84    | 0.75    | 0.81    |
| BERT-45M           | 0.84          | 0.74          | 0.85          | 0.86       | 0.78       | 0.79       | 0.85    | 0.76    | 0.82    |
| DMoEBERT-23M       | 0.86          | 0.78          | 0.84          | 0.87       | 0.77       | 0.84       | 0.86    | 0.77    | 0.84    |
| DMoEBERT-45M       | 0.87          | 0.79          | 0.85          | 0.87       | 0.78       | 0.86       | 0.87    | 0.79    | 0.86    |
| DMoEBERT-97M       | 0.88          | 0.78          | 0.86          | 0.85       | 0.80       | 0.86       | 0.87    | 0.79    | 0.86    |
| BERT-base (LoRA)   | 0.89          | 0.85          | 0.88          | 0.88       | 0.83       | 0.90       | 0.89    | 0.84    | 0.89    |
| RoBERTa-base (LoRA)| 0.90          | 0.85          | 0.91          | 0.90       | 0.86       | 0.91       | 0.90    | 0.86    | 0.91    |
-------------------------------------------------------------------------------------------------------------------------------------------
```

#### Per-class performance on QQP (Validation Set)
```
| Model          | Class          | Precision | Recall | F1-score |
|----------------|----------------|-----------|--------|----------|
| BERT-23M       | Not Duplicate  | 0.8676    | 0.8344 | 0.8507   |
| BERT-23M       | Duplicate      | 0.7326    | 0.7807 | 0.7559   |
| DMoEBERT-23M   | Not Duplicate  | 0.8991    | 0.8402 | 0.8687   |
| DMoEBERT-23M   | Duplicate      | 0.7528    | 0.8376 | 0.7930   |
*Validation set results (test labels not public).*
```

#### Per-class performance on SST-2 (Validation Set)
```
| Model          | Class     | Precision | Recall | F1-score |
|----------------|-----------|-----------|--------|----------|
| BERT-23M       | Negative  | 0.8153    | 0.7944 | 0.8047   |
| BERT-23M       | Positive  | 0.8066    | 0.8266 | 0.8165   |
| DMoEBERT-23M   | Negative  | 0.8134    | 0.7944 | 0.8038   |
| DMoEBERT-23M   | Positive  | 0.8062    | 0.8243 | 0.8151   |
*Validation set results (test labels not public).*
```

#### Per-class performance on AG News (Test Set)
```
| Model          | Class      | Precision | Recall | F1-score |
|----------------|------------|-----------|--------|----------|
| BERT-23M       | World      | 0.9510    | 0.8895 | 0.9192   |
| BERT-23M       | Sports     | 0.9557    | 0.9774 | 0.9664   |
| BERT-23M       | Business   | 0.8675    | 0.9026 | 0.8847   |
| BERT-23M       | Sci/Tech   | 0.8965    | 0.8979 | 0.8972   |
| DMoEBERT-23M   | World      | 0.9457    | 0.8979 | 0.9212   |
| DMoEBERT-23M   | Sports     | 0.9446    | 0.9863 | 0.9650   |
| DMoEBERT-23M   | Business   | 0.8844    | 0.8858 | 0.8851   |
| DMoEBERT-23M   | Sci/Tech   | 0.8910    | 0.8953 | 0.8931   |
```

## Routing Statistics Analysis
DyT exhibits substantially more balanced expert routing, with ~20× lower utilization variance and higher, more stable routing entropy compared to LayerNorm.
(Performed on QQP validation set)
```
| Metric               | **DyT (DMoEBERT)** | **LayerNorm (DMoEBERT)** |
| -------------------- | ------------------ | ------------------------ |
| Utilization Variance |     0.000004       | 0.000087                 |
| Routing Entropy      |     0.6527         | 0.5868                   |
| Entropy Std. Dev.    |     0.0594         | 0.1379                   |
```
#### Per-Expert Usage Distribution
```
| Expert | **DyT Usage (%)** | **DyT Tokens** | **LN Usage (%)** | **LN Tokens** |
| -----: | ----------------- | -------------- | ---------------- | ------------- |
|      0 | 16.73%            | 77,453         | 17.08%           | 79,074        |
|      1 | 16.96%            | 78,498         | 15.00%           | 69,414        |
|      2 | 16.34%            | 75,629         | 16.97%           | 78,525        |
|      3 | 16.50%            | 76,392         | 17.49%           | 80,957        |
|      4 | 16.70%            | 77,303         | 17.60%           | 81,455        |
|      5 | 16.76%            | 77,573         | 15.86%           | 73,423        |
```

## Analysis of Statistical Significance of Ablation Results Across Random Seeds

#### Per seed accuracy
```
| Model Variant        | Seed | Accuracy | Correct / Total |
| -------------------- | ---- | -------- | --------------- |
| **DMoEBERT-23M**     | 123  | 91.50%   | 6954 / 7600     |
|                      | 42   | 91.39%   | 6946 / 7600     |
|                      | 1337 | 91.63%   | 6964 / 7600     |
|----------------------|------|----------|-----------------|
| **DyT → LN**         | 123  | 91.22%   | 6933 / 7600     |
|                      | 42   | 91.34%   | 6942 / 7600     |
|                      | 1337 | 91.55%   | 6958 / 7600     |
|----------------------|-----------------|-----------------|
| **SparseMoE → FFWD** | 1337 | 91.43%   | 6949 / 7600     |
|                      | 42   | 91.18%   | 6930 / 7600     |
|                      | 123  | 91.50%   | 6954 / 7600     |
------------------------------------------------------------
```

#### Aggregate Statistics (Mean ± Std)
```
| Model Variant    | Accuracy (Mean ± Std) |
| ---------------- | --------------------- |
| **DMoEBERT-23M** | 91.51% ± 0.10%        |
| DyT → LN         | 91.37% ± 0.14%        |
| SparseMoE → FFWD | 91.37% ± 0.14%        |
```
#### Statistical Significance (paired t-test across seeds)
```
| Comparison               | Mean Δ | p-value | Significant (p < 0.05) |
| ------------------------ | ------ | ------- | ---------------------- |
| Main vs DyT → LN         | +0.14% | 0.1988  | ❌ No                   \
| Main vs SparseMoE → FFWD | +0.14% | 0.1837  | ❌ No                   |
```
controlled ablations were performed replacing Dynamic Tanh with LayerNorm and sparse MoE layers with dense feed-forward layers to isolate their effects. Each setting was evaluated over three matched random seeds (42, 123, 1337) on AG News. Across seeds, both ablations exhibit low variance (std.\ $\approx 0.10$--$0.14\%$) and mean accuracies comparable to the baseline. 

Paired t-tests confirm that neither ablation yields statistically significant differences relative to the main model (main vs DyT→LN: p = 0.199; main vs MoE→FFWD: p = 0.184), indicating that the primary contributions of DyT and sparse MoE lie in training stability and computational efficiency rather than final accuracy gains.

