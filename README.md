# thesis_88

DMoEBERT, is my thesis work. I replaced layer norm with META's Dynamic-Tanh to overcome normalization layer's computational overhead and feed-forward MLP layer with Sparsely activated mixture of experts with noisy top-k routing for efficiency.
I trained the complete pipeline from scratch. DMoEBERT outperforms BERT-base(110M) and RoBERTA-base(125M) in efficiency.

I compared my model against finetuned BERT-base(110M) and RoBERTA-base(125M). For finetuning, i used hugging face framework PEFT/LoRA.

I also compared my models against BERT-23M and BERT-45M and my models outperform those models both in accuracy and efficiency.

#### Rad the detailed report of my thesis [here](https://github.com/shovonSharma/thesis_88/blob/main/Report%20on%20DMoEBERT.pdf).

### ARCHITECTURE
![DMoEBERT Architecture](https://github.com/shovonSharma/thesis_88/blob/main/DMoEBERT.jpg)

### DATASET EXAMPLE

#### 🟩 SNLI Example (Stanford Natural Language Inference)

Premise : A man inspects the uniform of a figure in some East Asian country.

Hypothesis : The man is sleeping.

Label : Contradiction

Explanation : The premise describes a man actively inspecting something — contradicts “sleeping.”

#### 🟦 MNLI Example (Multi-Genre Natural Language Inference)

Premise (from "Government" genre) : The prime minister said the country will increase investment in renewable energy sources.

Hypothesis : The country plans to spend more money on wind and solar power.

Label : Entailment

Explanation : Investing more in renewable energy entails spending more on wind and solar power (common renewable types).

#### 🟥 ANLI Example (Adversarial NLI)

Premise : The teacher warned the students that if they didn’t stop talking, they’d miss their recess.

Hypothesis : The students stayed quiet and went out for recess.

Label : Contradiction

Explanation : The premise implies they were talking and being warned — contradicting that they stayed quiet and got recess.

#### 🟥 QQP Example (Quora Question Pairs)

Question 1: How can I improve my English speaking skills?

Question 2: What are the best ways to get better at speaking English?

Label: Duplicate

Explanation: Both questions ask for methods to improve spoken English, so they have the same meaning.

#### 🟥 SST-2 Example (Stanford Sentiment Treebank v2)

Sentence: The movie had great acting but a painfully slow storyline.

Label: Negative

Explanation: Despite praising the acting, the sentence expresses an overall negative opinion due to the slow storyline

#### 🟥 AG-NEWS Example (Quora Question Pairs)

Text: NASA announced a new mission to explore the surface of Mars using advanced robotics.

Label: Sci/Tech

Explanation: The text discusses a space mission and advanced technology, which falls under the Science/Technology category.

### Model configuration of DMoEBERT (DM) variants
![config](https://github.com/shovonSharma/thesis_88/blob/main/Model%20config.jpg)

### Performance comparison and ablation study on SNLI and ANLI test sets. Acc.=Accuracy, F1-M=F1-macro
![performance](https://github.com/shovonSharma/thesis_88/blob/main/performace_comparison.jpg)

### Per-class performance on SNLI test set. Ent. = Entailment, Neu. = Neutral, Con. = Contradiction.
![per-class performance](https://github.com/shovonSharma/thesis_88/blob/main/Per-class%20performance%20on%20SNLI%20test%20set.jpg)
