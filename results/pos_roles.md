# pos and roles

from src/explore_pos.py. numbers on the fixed 500 test queries unless said otherwise. pos source used: csv column.

| finding | value |
|---|---|
| queries | 500 |
| pos_source | csv column |
| pct_pos_column_filled | 97.91 |
| top_pos_tags | {'n.': 81092, 'a.': 34596, 'v. t.': 17977, 'v. i.': 7088, 'imp. & p. p.': 5778, 'p. pr. & vb. n.': 5770, 'adv.': 4447, 'pl. ': 3422, 'n. pl.': 1479, 'superl.': 1296, 'v.': 840, 'v. t. & i.': 438, 'prep.': 310, 'imp.': 221, 'p. p.': 202} |
| gold_pos_distribution | {'noun': 278, 'adj': 119, 'verb': 83, 'missing': 6, 'other': 6, 'adv': 5, 'partic': 2, 'interj': 1} |
| pct_words_whose_senses_differ_in_pos | 9.86 |
| headword_suffix_coverage_pct | 38.51 |
| headword_class_distribution | {'other': 65816, 'adj': 16739, 'noun_thing': 10541, 'agent': 8976, 'verb': 3602, 'place': 1358} |
| spacy_used | True |
| spacy_vs_column_agree_pct | 52.66 |
| pct_queries_with_a_cue | 31.6 |
| cue_matches_gold_pos_pct | 95.57 |
| cue_matches_gold_suffix_pct | 89.87 |
| pct_lost_by_hard_filter | 4.43 |
| agent_cue_queries | 17 |
| pct_agent_cue_gold_is_noun | 100.0 |
| pct_agent_cue_headword_is_agentive | 88.24 |
| pct_agent_role_headword_is_agentive | 87.5 |
| pct_nouns_that_are_agentive | 17.02 |
| patient_or_instrument_cue_queries | 2 |
| role_counts_queries | {'none': 364, 'action': 96, 'quality': 17, 'agent': 16, 'location': 5, 'instrument': 2} |
| role_counts_corpus | {'none': 129000, 'action': 29173, 'quality': 5782, 'agent': 5017, 'location': 770, 'instrument': 360, 'patient': 38} |
| pct_corpus_with_a_role | 24.18 |
| pct_multiword_answers | 0.6 |

notes

- the POS column is not empty, 97.9% of rows have a tag in it. our EDA read it as empty because the check we ran only accepted a single letter and this file writes them as Webster abbreviations, `n.`, `v. t.`, `prep.`, `a.`. so tags come from the file when it is there and from the ending when not.
- 273 distinct tag strings in the file, top ones: n. (81092), a. (34596), v. t. (17977), v. i. (7088), imp. & p. p. (5778), p. pr. & vb. n. (5770).
- 9.9% of headwords have senses with more than one POS in this corpus. so POS is a per sense property here, which is one more reason the feature goes next to the model and not as a filter over the whole word.
- headword endings still name a class for 38.5% of the 107,032 vocabulary. kept because that is the morphology channel, not because it beats the column.
- spaCy and the file agree on 52.7% of the sampled headwords. where they differ the file is usually right about the sense and the tagger is guessing on archaic 1913 text, so the tagger is a cross check and not the source.
- 31.6% of queries open with a cue that states the answer type. the cue matches the gold POS 95.6% of the time. a hard filter on that would have lost the gold answer for 4.4% of them, which is why the class stays a feature and a rerank bonus.
- on the astronomer question: of the 17 "person who" queries, 100.0% of the gold answers are nouns. astronomy is a noun too, so POS cannot separate them. the ending can: 88.2% of those gold words are agentive and only 17.0% of all nouns in the file are. that is the whole argument for the morphology channel.
- roles over the 500 queries: none 364, action 96, quality 17, agent 16, location 5, instrument 2. 24.2% of the whole corpus opens with one of these phrases, so the role is written in the text and needs no parser.
- 0.6% of the test answers are multi word entries which no POS filter can handle. they stay in the index and count as a limitation.
