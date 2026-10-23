# pos and roles

from src/explore_pos.py. numbers on the fixed 500 test queries unless said otherwise. pos source used: csv column.

| finding | value |
|---|---|
| queries | 500 |
| pos_source | csv column |
| pct_pos_column_filled | 97.63 |
| top_pos_tags | {'n.': 70786, 'a.': 26687, 'v. t.': 12505, 'v. i.': 5214, 'adv.': 3913, 'imp. & p. p.': 3189, 'p. pr. & vb. n.': 2877, 'pl. ': 2568, 'n. pl.': 1404, 'superl.': 631, 'v.': 611, 'v. t. & i.': 390, 'prep.': 219, 'imp.': 194, 'p. p.': 181} |
| gold_pos_distribution | {'noun': 288, 'verb': 117, 'adj': 72, 'adv': 9, 'other': 8, 'missing': 4, 'partic': 1, 'interj': 1} |
| pct_words_whose_senses_differ_in_pos | 7.09 |
| headword_suffix_coverage_pct | 40.03 |
| headword_class_distribution | {'other': 57147, 'adj': 14778, 'noun_thing': 10004, 'agent': 8692, 'verb': 3417, 'place': 1248} |
| spacy_used | True |
| spacy_vs_column_agree_pct | 55.81 |
| pct_queries_with_a_cue | 34.8 |
| cue_matches_gold_pos_pct | 90.8 |
| cue_matches_gold_suffix_pct | 90.23 |
| pct_lost_by_hard_filter | 6.9 |
| agent_cue_queries | 16 |
| pct_agent_cue_gold_is_noun | 87.5 |
| pct_agent_cue_headword_is_agentive | 81.25 |
| pct_agent_role_headword_is_agentive | 80.0 |
| pct_nouns_that_are_agentive | 16.87 |
| patient_or_instrument_cue_queries | 1 |
| role_counts_queries | {'none': 340, 'action': 126, 'agent': 15, 'quality': 13, 'location': 5, 'instrument': 1} |
| role_counts_corpus | {'none': 105114, 'action': 20624, 'agent': 4739, 'quality': 4454, 'location': 653, 'instrument': 297, 'patient': 26} |
| pct_corpus_with_a_role | 22.66 |
| pct_multiword_answers | 0.6 |

notes

- the POS column is not empty, 97.6% of rows have a tag in it. our EDA read it as empty because the check we ran only accepted a single letter and this file writes them as Webster abbreviations, `n.`, `v. t.`, `prep.`, `a.`. so tags come from the file when it is there and from the ending when not.
- 248 distinct tag strings in the file, top ones: n. (70786), a. (26687), v. t. (12505), v. i. (5214), adv. (3913), imp. & p. p. (3189).
- 7.1% of headwords have senses with more than one POS in this corpus. so POS is a per sense property here, which is one more reason the feature goes next to the model and not as a filter over the whole word.
- headword endings still name a class for 40.0% of the 95,286 vocabulary. kept because that is the morphology channel, not because it beats the column.
- spaCy and the file agree on 55.8% of the sampled headwords. where they differ the file is usually right about the sense and the tagger is guessing on archaic 1913 text, so the tagger is a cross check and not the source.
- 34.8% of queries open with a cue that states the answer type. the cue matches the gold POS 90.8% of the time. a hard filter on that would have lost the gold answer for 6.9% of them, which is why the class stays a feature and a rerank bonus.
- on the astronomer question: of the 16 "person who" queries, 87.5% of the gold answers are nouns. astronomy is a noun too, so POS cannot separate them. the ending can: 81.2% of those gold words are agentive and only 16.9% of all nouns in the file are. that is the whole argument for the morphology channel.
- roles over the 500 queries: none 340, action 126, agent 15, quality 13, location 5, instrument 1. 22.7% of the whole corpus opens with one of these phrases, so the role is written in the text and needs no parser.
- 0.6% of the test answers are multi word entries which no POS filter can handle. they stay in the index and count as a limitation.
