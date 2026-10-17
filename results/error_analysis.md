# failures, bm25 leave_one_out

goes with results/baselines.md. 484 of 500 queries miss at rank 1 here.

categories, from the script:

- unreachable: 213
- outranked: 120
- no lexical overlap: 100
- right word, deep: 50
- tied definition: 1

two of these are not the model failing. unreachable means the word had one row and that row is the query, so removing it deletes the answer. tied definition means two words share the string, so rank 1 is a coin flip no matter what the ranker is. both are properties of the file.

three things the categories do not say on their own

- 14.5% of these misses have a rank 1 that is a word the definition itself names as a synonym. so the model is often not lost, it returned a different dictionary word for the same thing and the strict metric calls it wrong. that is the same argument as the tolerant column, just counted on the failures.
- 116 queries had the gold word inside the top 100 but not at rank 1, median rank 13. that is the pool a dense model or a reranker has to reorder, and it is why R@100 of 0.264 and R@1 of 0.032 are not the same statement.
- on the misses the opening cue disagrees with the tag in the file for 1.4%, which is the same rule pos_roles.md uses, computed there over all 500 queries and here over the failures only.

21 failures to read by hand, 5 from each category

| query | gold | rank 1 | rank 2 | category | cue | gold pos | rank of gold | what happened |
|---|---|---|---|---|---|---|---|---|
| The Arabic definite article answering to the English the; as... | **Al-** | Definite | Print | no lexical overlap | none | adj | not in top 100 |  |
| The place for baking bread; a bakehouse. | **Bakery** | Baking | Pattypan | no lexical overlap | place | noun | not in top 100 |  |
| The woody portion flax  which is separated from the fiber as... | **Boon** | Scutch | Scutch | no lexical overlap | none | noun | not in top 100 |  |
| A sunk panel of ceilings or soffits. | **Caisson** | Paneled | Panelled | no lexical overlap | none | noun | not in top 100 |  |
| A separating tissue; a partition; a septum. | **Dissepiment** | Septum | Elocular | no lexical overlap | none | noun | not in top 100 |  |
| Befitting an athlete; strong; muscular; robust; vigorous; as... | **Athletic** | Robust | Stout | outranked | none | adj | 111 |  |
| To abrade or rub off any outer covering from; as to bark one... | **Bark** | Abrade | Cortex | outranked | verb | verb | 30 |  |
| To dispose in proper form; to reduce to order; to put in pro... | **Compose** | Adjust | Set | outranked | verb | verb | 329 |  |
| Full of grief; sad; sorrowful; doleful; dismal; as  a doloro... | **Dolorous** | Doleful | Dolesome | outranked | none | adj | 126 |  |
| To enlarge the bore of a gun barrel by drawing  instead of t... | **Drawbore** | Bore | Couching | outranked | verb | verb | 649 |  |
| With the hand turned backward; as  a backhanded blow. | **Backhanded** | Backhander | Reverse | right word, deep | none | adj | 5 |  |
| To follow by dodging  or suddenly shifting from place to pla... | **Dodge** | Shifting | Veering | right word, deep | verb | verb | 6 |  |
| Any soft woolly covering resembling a fleece. | **Fleece** | Woolly | Carpet | right word, deep | none | noun | 4 |  |
| A fret or chafe  as in cloth; a place injured by rubbing. | **Fray** | Chafe | Frote | right word, deep | none | noun | 5 |  |
| To make an opening in; to breach. | **Gap** | Breach | Rent | right word, deep | verb | verb | 5 |  |
| To repay; to requite. | **Quittance** | Foryelde | Cope | tied definition | verb | verb | not in top 100 |  |
| The act of abolishing; abolition; destruction. | **Abolishment** | Abolition | Annulment | unreachable | noun_thing | noun | not in top 100 |  |
| The houseleek (Sempervivum tectorum). | **Ayegreen** | Sempervivum | Sempervive | unreachable | none | noun | not in top 100 |  |
| The governor of a province of the Ottoman empire  next in di... | **Beglerbeg** | Vizier-azem | Nabob | unreachable | none | noun | not in top 100 |  |
| Equivalent in combining or displacing power to two atoms of ... | **Bivalent** | Dyad | Equivalence | unreachable | none | partic | not in top 100 |  |
| The state of being bloated. | **Bloatedness** | Bloat | Emphysematous | unreachable | noun_thing | noun | not in top 100 |  |
