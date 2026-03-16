# Relevant References for Dataset Construction for RAG Evaluation

## 1. TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages

**Authors:** Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, Jennimaria Palomaki  
**Published:** *Transactions of the Association for Computational Linguistics*, Vol. 8, pp. 454–470, 2020  
**Links:** [arXiv:2003.05002](https://arxiv.org/abs/2003.05002) · [ACL Anthology](https://aclanthology.org/2020.tacl-1.30/)

### Summary and Key Findings

TyDi QA is a large-scale multilingual question answering benchmark covering 11 languages with 204K question-answer pairs.
- Main methodological contribution is its **information-seeking design**: questions are written by annotators who genuinely want to know the answer and do not yet know it, avoiding the *priming effect* present in most prior QA datasets (where annotators write questions after reading the passage). 
- Data is collected directly in each target language (not through translation) ensuring linguistic and cultural authenticity.
- The dataset defines two evaluation tasks: 
    - **passage selection**: identifying the relevant paragraph in a Wikipedia article
    - **minimal answer extraction**: finding the shortest correct answer text snippet.
- The **unanswerable category** where no passage in the article contains the answer is a core feature of the benchmark and one of its most significant contributions to robust evaluation design.

### How to apply to our dataset

| TyDi QA concept | Application in this dataset |
|---|---|
| Information-seeking question design (no priming) | Questions authored/generated without the annotator looking at a specific chunk first |
| Unanswerable question category | `question_type: out_of_scope` to measure whether the chatbot abstains correctly |
| Gold passage annotation | `reference_contexts` field to allow `context_recall` computation |
| Stratified evaluation by question type | Score breakdown by `question_type` (factual, procedural, comparative, out_of_scope, ambiguous) |
| Inter-annotator agreement reporting | Cohen's κ reported on a 20% double-annotated subset |

---

## 2. ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems

**Authors:** Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia  
**Published:** arXiv preprint, first released November 2023; updated March 2024  
**Venue:** North American Chapter of the Association for Computational Linguistics (NAACL)  
**Links:** [arXiv:2311.09476](https://arxiv.org/abs/2311.09476) · [GitHub](https://github.com/stanford-futuredata/ARES)

### Summary & Key Findings

ARES introduces a fully automated pipeline for evaluating RAG systems without requiring large-scale human annotation. The framework addresses a core bottleneck in RAG evaluation: hand-annotating queries, retrieved passages, and reference answers is expensive, slow, and brittle **under domain shift**. ARES solves this through a three-stage pipeline:

1. **Synthetic dataset generation.** An LLM (originally FLAN-T5 XXL) is prompted to generate question-answer pairs from passages in the target corpus. A quality filter discards queries where the generated question cannot retrieve its source passage as the top result — ensuring only coherent, grounded questions enter the evaluation set.
2. **LLM judge fine-tuning.** Lightweight classifier models are fine-tuned on the synthetic data to act as judges for three evaluation dimensions: *context relevance*, *answer faithfulness*, and *answer relevance*. Negative examples are generated via two strategies: weak negatives (random in-domain passages) and strong negatives (answers generated from unrelated passages).
3. **Prediction-powered inference (PPI).** A small set of ~150 human-annotated examples is used to calibrate the judges and produce statistically grounded confidence intervals for all scores, correcting for the noise introduced by LLM-generated labels.

- **Synthetic datasets can replace large-scale human annotation.** ARES achieves accurate RAG system rankings using only ~150 human-annotated examples for calibration, compared to the thousands typically required for traditional benchmarks.
- **LLM-generated labels introduce noise that must be corrected.** Without PPI calibration, synthetic judges exhibit **systematic biases**. The paper provides concrete evidence that the ~150-example human validation set is necessary — below ~100 examples, ranking accuracy degrades meaningfully.
    - Inter-Annotator Agreement (IAA): used to validate human judgments of RAG performance 
    - Prediction-Powered Inference (PPI): automated technique that uses small amount of human-labeled data to caliibrate LLM-based judges
- **Domain transfer works with minimal re-annotation.** Judges trained on one domain (e.g., Wikipedia-based KILT tasks) remain effective when evaluated on a different domain, requiring only a new ~150-example validation set rather than full re-annotation.
- **Filtering synthetic queries by retrieval quality.** Queries that cannot retrieve their source passage as the top result are low-quality and should be discarded. This filtering step is a key practical contribution with direct implications for how synthetic evaluation sets should be constructed.

### How to apply to our dataset

| ARES concept | Application in this dataset |
|---|---|
| LLM-generated synthetic questions from corpus passages and noting data origin | Supports this generation methodology but emphasizes tracking the `generation_method: llm_generated` and `llm_then_human_verified` field values |
| Three evaluation dimensions (context relevance, answer faithfulness, answer relevance) | Already map to RAGAS/DeepEval metrics.|
| Quality filtering of synthetic queries | Discarding LLM-generated questions that fail a retrieval round-trip check (setting a threshold on retrieval metrics). |
| Small human validation set for calibration | Annotating a subset of system-answered questions can be used for inter-annotator agreement and PPI to compute the systematic LLM-judge error and produce a confidence interval on the correct judge score |

---

## 3. Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems

**Authors:** Rafael Teixeira de Lima, Shubham Gupta, Cesar Berrospi, Lokesh Mishra, Michele Dolfi, Peter Staar, Panagiotis Vagenas  
**Published:**arXiv preprint, submited November 2024, to be published in the 31st International Conference on Computational Linguistics (COLING 2025)
**Links:** [arXiv:2411.19710](https://arxiv.org/abs/2411.19710)

### Summary & Key Findings

- Domain-specific RAG system evaluation on speicifc corpus (generic benchmarks cannot cover it)
- Proposes a RAG-specific dataset taxonomy based on characterizing Q&A datasets through labels and shows that common RAG dataset generation tools produce unbalanced data across question types.
- [context, query] pairs classified by the **type of answer** the context provides to the query covering categories like fact_single, summary
- Creates a schema from seed documents defining the structure and uses it to generate synthetic documents to have a more structured, auditable steps instead of a single black-box generation call. More diverse samples than zero/one-shot prompting. 
- Completeness, Hallucination and Irrelevance metrics - LLM evaluated
- Fact single, summary and reasoning type of questions.

### How to apply to our dataset

| RAGEval concept | Application in this dataset |
|---|---|
| Structured Generation Process | Methodology can be applied to get broader type of pairs q&A and reference context. Also help in the classification of question types. |
| Stratified Evaluation | Characterizing Q&A Datasets (question type -> based on answer type) fact_single, summary and reasoning type|
| Completeness, Hallucination and Irrelevance metrics | Somehow map to RAGAS/DeepEval metrics.|


## 4. RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework

**Authors:** Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan, Zhenghao Liu, Shi Yu, Ruobing Wang, Shuo Wang, Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, Maosong Sun
**Published:** Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, July 2025
**Links:** [ACL Anthology](https://aclanthology.org/2025.acl-long.418/)

### Summary & Key Findings

They introduce RAGEval framework for generating scenario-specific datasets to evalaute RAG systems. Prioritize factual accuracy and scenario-specific knowledge. 
- They introduce quality assesment with human verification of the generated dataset and evaluation. 
- The evaluation seems manual with their own custom metrics, include Hallucination Aspect on cases where retrieval is not complete. 
- They consider balance and diversity over the question types, proposing a stratified evaluation. 

Single-Document QA
| Question Type | Definition |
|---|---|
|Factual | Questions targeting specific details to test RAG's retrieval accuracy|
| Summarization | Quetions thaat require comprhensive answers covering all relevant information (recall rate of retrieval)|
| Multi-hop Reasoning | Questions involving logical relationships among detail/events in the document, to assess reasoning ability |

Multi-Document QA
| Question Type | Definition |
|---|---|
|Information Integration | Questions needing information from combining 2 documents with distinct information fragments|
| Numerical Comparison | Questions requiring to find and compare data fragments |
| Temporal Sequence | Determining cronological order of events, testing reasoning skills|

> Unanswerable Questions: no corresponding informaion fragment exists or is insufficient for an answer

### How to apply to our dataset

| RAGEval concept | Application in this dataset |
|---|---|
| Quality Assesment with human verification | Also supports this generation methodology but emphasizes tracking the `generation_method: llm_generated` and `llm_then_human_verified` field values |
| Hallucination Metric | Identify content contradictions still evaluating cases where retrieval is not complete. |
| Stratified Evaluation | Characterizing Q&A Datasets through labels and balancing the quantity of each type. |



## Citation

```bibtex
@article{clark-etal-2020-tydi,
  title     = {{T}y{D}i {QA}: A Benchmark for Information-Seeking Question Answering
               in Typologically Diverse Languages},
  author    = {Clark, Jonathan H. and Choi, Eunsol and Collins, Michael and
               Garrette, Dan and Kwiatkowski, Tom and Nikolaev, Vitaly and
               Palomaki, Jennimaria},
  journal   = {Transactions of the Association for Computational Linguistics},
  volume    = {8},
  pages     = {454--470},
  year      = {2020},
  publisher = {MIT Press},
  url       = {https://arxiv.org/abs/2003.05002}
}

@misc{saadfalcon2023ares,
  title         = {ARES: An Automated Evaluation Framework for
                   Retrieval-Augmented Generation Systems},
  author        = {Saad-Falcon, Jon and Khattab, Omar and Potts, Christopher
                   and Zaharia, Matei},
  year          = {2023},
  eprint        = {2311.09476},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2311.09476}
}

@inproceedings{teixeira-de-lima-etal-2025-know,
  title     = {Know Your {RAG}: Dataset Taxonomy and Generation Strategies
               for Evaluating {RAG} Systems},
  author    = {Teixeira de Lima, Rafael and Gupta, Shubham and
               Berrospi Ramis, Cesar and Mishra, Lokesh and
               Dolfi, Michele and Staar, Peter and Vagenas, Panagiotis},
  booktitle = {Proceedings of the 31st International Conference on
               Computational Linguistics: Industry Track},
  pages     = {39--57},
  year      = {2025},
  month     = jan,
  address   = {Abu Dhabi, UAE},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2025.coling-industry.4/}
}

@inproceedings{zhu-etal-2025-rageval,
  title     = {{RAGE}val: Scenario Specific {RAG} Evaluation Dataset
               Generation Framework},
  author    = {Zhu, Kunlun and Luo, Yifan and Xu, Dingling and
               Yan, Yukun and Liu, Zhenghao and Yu, Shi and
               Wang, Ruobing and Wang, Shuo and Li, Yishan and
               Zhang, Nan and Han, Xu and Liu, Zhiyuan and Sun, Maosong},
  booktitle = {Proceedings of the 63rd Annual Meeting of the
               Association for Computational Linguistics
               (Volume 1: Long Papers)},
  pages     = {8520--8544},
  year      = {2025},
  month     = jul,
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2025.acl-long.418/}
}
```