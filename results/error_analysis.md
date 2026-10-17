# failures, bm25 leave_one_out

goes with results/baselines.md. 484 of 500 queries miss at rank 1 here.

categories, from the script:

- unreachable: 213
- outranked: 120
- no lexical overlap: 100
- right word, deep: 50
- tied definition: 1

two of these are not the model failing. unreachable means the word had one row and that row is the query, so removing it deletes the answer. tied definition means two words share the string, so rank 1 is a coin flip no matter what the ranker is. both are properties of the file.

first 20 failures, to read by hand

| query | gold | rank 1 | rank 2 | category | cue | gold class | what happened |
|---|---|---|---|---|---|---|---|
| superl. of Able. | **Abler** | Biggest | Ferrest | right word, deep | none | agent |  |
| The act of abolishing; abolition; destruction. | **Abolishment** | Abolition | Annulment | unreachable | noun_thing | noun_thing |  |
| In a plane  the two imaginary circular points at infinity; i... | **Absolute** | Hyperspace | Circuline | no lexical overlap | none | other |  |
| Incomprehensibility of things; the doctrine held by the anci... | **Acatalepsy** | Preexistence | Euhemerism | unreachable | none | other |  |
| Shaped like a shallow cup; saucer-shaped; as  an acetabulifo... | **Acetabuliform** | Acetabular | Chaliced | unreachable | none | other |  |
| Congenitally united with an organ of another kind  as calyx ... | **Adherent** | Inferior | Perigynous | no lexical overlap | none | adj |  |
| One who adheres; one who adheres; one who follows a leader  ... | **Adherent** | Platonist | Calvinist | outranked | agent | adj |  |
| Entitled to be admitted  or worthy of being admitted; that m... | **Admissible** | Permissible | Allowable | unreachable | none | adj |  |
| Violent contest or striving. | **Agony** | Contention | Affectation | no lexical overlap | none | other |  |
| A cavity in the cellular tissue of plants  containing air on... | **Air cell** | Emphysema | Sclerema | outranked | none | other |  |
| The Arabic definite article answering to the English the; as... | **Al-** | Definite | Print | no lexical overlap | none | other |  |
| To keep in excitement; to disturb. | **Alarm** | Sturb | Commove | no lexical overlap | verb | other |  |
| Nourishing matter stored up within the integuments of the se... | **Albumen** | sarcoderma | Grain | no lexical overlap | none | other |  |
| An instrument to ascertain the strength of alkalies  or the ... | **Alkalimeter** | Alkaline | Alkalimetry | unreachable | none | agent |  |
| A subsulphate of alumina and potash; alunite. | **Alum stone** | Gismondite | Potashes | unreachable | none | other |  |
| The poisonous principle of some fungi. | **Amanitine** | Antiarin | Cicutoxin | unreachable | none | other |  |
| Without crystallization in the ultimate texture of a solid s... | **Amorphous** | Crystalline | Imporous | no lexical overlap | none | adj |  |
| A race of giants living in Palestine. | **Anaks** | Giantry | Asura | unreachable | none | other |  |
| Overthrowing; defeating; -- applied to Plato's refutative di... | **Anatreptic** | Deictic | Elenctical | unreachable | none | adj |  |
| In English; in the English manner; as  Livorno Anglice Legho... | **Anglice** | Brogue | L | unreachable | none | other |  |
