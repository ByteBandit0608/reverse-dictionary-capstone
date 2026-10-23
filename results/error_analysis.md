# failures, bm25 leave_one_out

goes with results/baselines.md. 477 of 500 queries miss at rank 1 here.

categories, from the script:

- unreachable: 0
- no lexical overlap: 244
- outranked: 176
- right word, deep: 52
- tied definition: 5

two of these are properties of the file and not of the model. unreachable counts a query whose word had only the one row the protocol removes, it is 0 here, and the reason it is 0 is the rule that a query word keeps at least two rows. tied definition counts answers sharing their string with another word, 5 here, where rank 1 is a coin flip whatever the ranker is.

three things the categories do not say on their own

- 6.5% of these misses have a rank 1 that is a word the definition itself names as a synonym. so the model is often not lost, it returned a different dictionary word for the same thing and the strict metric calls it wrong. that is the same argument as the tolerant column, just counted on the failures.
- 138 queries had the gold word inside the top 100 but not at rank 1, median rank 16. that is the pool a dense model or a reranker has to reorder, and it is why R@100 of 0.322 and R@1 of 0.046 are not the same statement.
- on the misses the opening cue disagrees with the tag in the file for 2.5%, which is the same rule pos_roles.md uses, computed there over the whole query set and here over the failures only.

phenomena, counted over the whole failure set instead of listed as examples

| what the model is actually missing | queries | mean tokens | mean rare terms |
|---|---|---|---|
| the dictionary words it up differently | 220 | 10.2 | 3% |
| right word found, buried past the top 10 | 162 | 12.6 | 2% |
| right word found, ranked just below | 47 | 16.1 | 2% |
| returned a synonym the definition itself names | 19 | 14.0 | 5% |
| long definition, several ideas at once | 18 | 30.8 | 4% |
| rare or technical wording in the query | 6 | 3.5 | 75% |
| structural, two words share the exact string | 5 | 4.6 | 20% |

how much the misses overlap with the other methods, same 500 queries

- all_mpnet_base_v2: 477 of its 500 misses are also bm25 misses, 95% of the union of the two sets. a high share means both methods go wrong on the same queries and a reranker over the two has little to add. a low one means they are looking at different things.

20 failures to read by hand, 5 from each category

| query | gold | rank 1 | rank 2 | category | cue | gold pos | rank of gold | what happened |
|---|---|---|---|---|---|---|---|---|
| One of the small grains or drupelets which make up some kind... | **Acinus** | Drupelet | Rubus | no lexical overlap | none | noun | not in top 100 |  |
| A shaft or missile intended to be shot from a crossbow or ca... | **Bolt** | Dart | Cock | no lexical overlap | none | noun | not in top 100 |  |
| An excrescence on a horse's frush or frog. | **Bouillon** | Frush | Chit | no lexical overlap | none | noun | not in top 100 |  |
| Having a broken back; as  a broken-backed chair. | **Broken-backed** | Razor-backed | Saddled | no lexical overlap | adj | adj | not in top 100 |  |
| To burn to a coal; to char. | **Chark** | Coal | Charred | no lexical overlap | verb | verb | not in top 100 |  |
| Fit or fitted; suited; suitable; appropriate. | **Apt** | Fitting | Idoneous | outranked | none | adj | 241 |  |
| A member of a court or council. | **Bencher** | Councilor | Councilman | outranked | none | noun | 330 |  |
| Having no depth; shallow. | **Depthless** | Shallow | Sinewless | outranked | adj | adj | 140 |  |
| The solid part of an animal's tail  as distinguished from th... | **Dock** | Clipper | Posteriors | outranked | none | noun | 776 |  |
| To obtain from some cause or origin; to infer from evidence ... | **Draw** | Deduce | Collect | outranked | verb | verb | 57 |  |
| Of or pertaining to Caledonia or Scotland; Scottish; Scotch. | **Caledonian** | Scotch | Caledonian | right word, deep | none | adj | 2 |  |
| In the United States navy  the lowest grade of officers in l... | **Midshipman** | Ensign | Court-martial | right word, deep | none | noun | 6 |  |
| To make a return in words or writing; to respond; to answer. | **Reply** | Return | Reanswer | right word, deep | verb | verb | 3 |  |
| To smile in an affected or conceited manner; to smile with a... | **Smirk** | Simper | Smirk | right word, deep | verb | verb | 2 |  |
| To wake; to cease to sleep; to be awakened. | **Waken** | Wake | Watch | right word, deep | verb | verb | 5 |  |
| The thread of a screw. | **Fillet** | Worm | Chase | tied definition | none | noun | not in top 100 |  |
| One who harries. | **Harrower** | Harrier | Conferrer | tied definition | agent | noun | 412 |  |
| One who speaks for another. | **Prolocutor** | Spokesman | Speaker | tied definition | agent | noun | not in top 100 |  |
| imp. of Sit. | **Sate** | Sat | Seten | tied definition | none | missing | 4 |  |
| The American night heron. See under Night. | **Squawk** | Qua-bird | Wapacut | tied definition | none | noun | not in top 100 |  |
