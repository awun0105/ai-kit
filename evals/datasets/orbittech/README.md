# OrbitTech Store sample dataset

10 customer-support policy documents (`corpus/`) and a 20-question golden set
(`golden.json`: easy, medium, hard and adversarial questions, each with the expected
answer and the source passages that support it).

From the author's solution to the AI20K D14 AI-Evaluation lab
(`K4-L3B-LamQuangAnhQuan-2A202602467-AIEvaluation`, branch `solution` at `f545662`:
`data/technology_store/`, `golden_dataset.json`). The corpus is synthetic.

Used by `scripts/eval_rag.py` as the default dataset and by the offline retrieval
regression gate in CI.
