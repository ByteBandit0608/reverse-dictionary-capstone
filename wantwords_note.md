# WantWords

## 1. Core idea

A plain neural reverse dictionary encodes the query and returns the words with the closest embeddings. This fails on low-frequency words (poor embeddings) and on words whose embeddings are close but whose properties contradict the query (for example, the wrong part of speech).

The multi-channel model adds several **predictors** (POS, morphology and sememes are the three covered in this note) that guess characteristics of the target word from the query. Each predictor is a channel, and a word's final score combines the embedding score with its score in every channel. The paper's example is "expressway": a person thinking of "a road where cars go very quickly" would guess it is a noun, belongs to an entity-like category, and probably contains the morpheme "way".

## 2. Architecture

- **Sentence encoder:** a BiLSTM with attention over pretrained word embeddings (GoogleNews word2vec, 300-d, frozen). Attention lets important words, such as the genus word of a definition, count more.
- **Main word score:** the sentence vector is projected into the word-embedding space, and each word is scored by dot product.
- **Channel scores:** each predictor reads the sentence vector (or the per-token hidden states) and gives each candidate word a score from the labels attached to that word.
- **Final score:** the word score plus the weighted sum of the channel scores. In the paper's English setup all weights are 1.
- **Loss:** one-versus-all cross-entropy.

## 3. The feature channels

### 3.1 POS tag (internal channel)
- **Source of labels:** the POS tags of the word's senses.
- **How it works:** the sentence vector goes through a single-layer classifier that scores every POS tag. A word's channel score is the sum of the scores of its POS tags. A noun-like description therefore boosts nouns and demotes verbs and adjectives.
- **English model detail:** the English model drops this predictor, because WordNet lexical names already contain POS (for example `noun.animal`). The Chinese model keeps a POS predictor.

### 3.2 Morphology (internal channel)
- **Source of labels:** morphemes (roots and affixes). In English the words are segmented with Morfessor. In Chinese each character is treated as a morpheme.
- **How it works:** every token's hidden state is scored against all morphemes, and the scores are max-pooled over the query. A word's channel score is the sum of the scores of its own morphemes.
- **Intuition:** query words align with word parts. In "expressway", "road" aligns with `way` and "quickly" with `express`. In "postnuptial", "after" aligns with `post`.
- **Effect:** it helps rare words, because their parts are common even when the whole word is rare.

### 3.3 Sememes (external channel)
- **Source of labels:** HowNet. A sememe is the minimum unit of meaning, and HowNet has about 2,000 of them. The English side covers 43,321 words, obtained through OpenHowNet. Structure and relations are dropped, so each word just has a set of sememe labels.
- **How it works:** same as the morpheme predictor. Each hidden state is scored against every sememe, then max-pooled, and a word's score is the sum over its own sememes.
- **Example:** "expressway" has the sememes route and fast, matching "road" and "quickly" in its definition.

### 3.4 Also in the paper (outside the three requested features)
The paper has one more predictor, **word category**, which uses WordNet lexical names (45 classes, such as `noun.animal`) to match the genus of the definition. It is mentioned here only because the English model uses it in place of a separate POS predictor, as the lexical names already carry POS. Our note focuses on POS, morphology and sememes.

## 4. What the paper reports

On the English test sets, each channel improved over a plain BiLSTM, and the combined model was best. On the human-written description set, acc@10 rose from about 0.60 (BiLSTM) to about 0.62 to 0.64 with a single extra channel, and to 0.64 with all of them (acc@100 0.83 to 0.88). Gains are larger for unseen and rare words. Giving the system the POS tag as extra prior knowledge helped only a little, which suggests the model already predicts POS well. Giving the initial letter or the word length helped far more.