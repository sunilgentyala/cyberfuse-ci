# CyberFuse-CI

**Adversarially Resilient Vulnerability Detection Through Heterogeneous Multi-Source Data Fusion and LLM-Augmented Reasoning**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![IEEE ICETCI 2026](https://img.shields.io/badge/IEEE-ICETCI%202026-orange.svg)](https://icetci.org)
[![Release](https://img.shields.io/badge/release-v1.0.0-brightgreen.svg)](https://github.com/sunilgentyala/cyberfuse-ci/releases)

This repository is the official companion artifact for the paper submitted to **ICETCI 2026** (Sixth International Conference on Emerging Techniques in Computational Intelligence), Mahindra University, Hyderabad, August 19-22, 2026.

> **Paper:** *CyberFuse-CI: Adversarially Resilient Vulnerability Detection Through Heterogeneous Multi-Source Data Fusion and LLM-Augmented Reasoning*
>
> **Authors:** Sunil Gentyala, Sunil Kumar Mudusu, Praveen Kumar Mannam, Sathish Allam, Rakesh Prakash

---

## What CyberFuse-CI Does

Enterprise vulnerability data arrives from sources that were never designed to communicate with each other. CVE records, STIX threat intelligence bundles, SIEM event logs, and source code repositories all describe security problems in different formats, different schemas, and at different levels of abstraction.

CyberFuse-CI fuses all four into a single adversarially hardened detection pipeline:

```
[ CVE / NVD ] ---+
[ STIX / TAXII ] -+---> [ Layer 1: Normalization ] ---> [ Layer 2: LLM Knowledge Graph ] ---> [ Layer 3: Adversarial Classifier + MITRE Mapping ]
[ SIEM Logs ] ---+
[ Code Repos ] --+
```

The framework maps every detected vulnerability to a MITRE ATT&CK technique and a remediation control, producing actionable output rather than raw scores.

---

## Key Results (CVEfixes + Enterprise SIEM corpus)

| Metric | CyberFuse-CI | Best Baseline |
|---|---|---|
| F1-Score | **94.7%** | 86.8% (GPT-4 zero-shot) |
| Adversarial Accuracy (FGSM) | **91.2%** | 71.3% (CodeBERT single-source) |
| Zero-Day Detection Lead Time | **6.2 days early** | 2.1 days (STIX-only) |
| False Positive Rate | **4.7%** | 11.2% |
| Detection Latency (per event) | **11.4 ms** | -- |

---

## Repository Structure

```
cyberfuse-ci/
├── connectors/                 # Layer 1: Multi-source ingestion and normalization
│   ├── nvd_connector.py        # CVE and NVD REST API v2.0 ingestion
│   ├── stix_connector.py       # STIX 2.1 / TAXII 2.1 bundle parser
│   ├── siem_normalizer.py      # Syslog and CEF log normalizer
│   ├── code_repo_connector.py  # CVEfixes dataset loader
│   └── entity_schema.py        # Shared intermediate entity tuple schema
│
├── knowledge_graph/            # Layer 2: LLM-augmented knowledge graph
│   ├── ner_extractor.py        # BiLSTM-CRF named entity recognizer
│   ├── relation_classifier.py  # Attention-based CNN relation extractor
│   ├── graph_builder.py        # OWL ontology-aligned knowledge graph builder
│   ├── gat_link_predictor.py   # Graph Attention Network for link prediction
│   └── llm_link_validator.py   # GPT-4 candidate link generator and validator
│
├── adversarial_training/       # Layer 3a: Adversarial hardening
│   ├── fgsm_attack.py          # Fast Gradient Sign Method perturbation
│   ├── adversarial_trainer.py  # Mixed clean + adversarial training harness
│   └── robustness_eval.py      # Clean vs adversarial accuracy reporter
│
├── mitre_mapping/              # Layer 3b: MITRE ATT&CK and ATLAS mapping
│   ├── attack_embedding.py     # ATT&CK v15 technique embedding index builder
│   ├── technique_mapper.py     # Nearest-neighbor technique lookup
│   └── mitigation_recommender.py # Mitigation control recommendation
│
├── evaluation/                 # Benchmark scripts
│   ├── run_benchmark.py        # Full pipeline evaluation on CVEfixes
│   ├── metrics.py              # Precision, recall, F1, adversarial accuracy
│   └── results/                # CSV outputs matching paper tables
│
├── ontology/
│   └── cyberfuse_ontology.owl  # OWL vulnerability ontology (ATT&CK aligned)
│
├── tests/                      # Unit and integration tests
│   ├── test_connectors.py
│   ├── test_knowledge_graph.py
│   ├── test_adversarial.py
│   └── test_mitre_mapping.py
│
├── docs/
│   └── architecture.md         # Detailed architecture documentation
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/sunilgentyala/cyberfuse-ci.git
cd cyberfuse-ci
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your API keys in environment variables before running:

```bash
export NVD_API_KEY="your_nvd_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here"
export TAXII_SERVER_URL="https://cti-taxii.mitre.org/taxii/"
```

You can get a free NVD API key at: https://nvd.nist.gov/developers/request-an-api-key

---

## Quick Start

### Run the full pipeline on CVEfixes

```bash
python evaluation/run_benchmark.py --dataset cvefixes --output evaluation/results/
```

### Ingest live CVE data

```bash
python connectors/nvd_connector.py --days 7 --output data/cve_feed.jsonl
```

### Run adversarial robustness evaluation

```bash
python adversarial_training/robustness_eval.py \
    --model checkpoints/xgboost_classifier.pkl \
    --epsilon 0.01 \
    --output evaluation/results/adversarial_report.csv
```

### Map a CVE to MITRE ATT&CK

```bash
python mitre_mapping/technique_mapper.py --cve CVE-2024-12345
```

---

## Datasets

| Dataset | Source | Access |
|---|---|---|
| CVEfixes | Bhandari et al., PROMISE 2021 | https://github.com/secureIT-project/CVEfixes |
| NVD CVE Feed | NIST NVD REST API v2.0 | https://nvd.nist.gov/developers/vulnerabilities |
| MITRE ATT&CK STIX | MITRE ATT&CK v15 | https://github.com/mitre/cti |
| CICIDS2017 (optional) | Canadian Institute for Cybersecurity | https://www.unb.ca/cic/datasets/ids-2017.html |

The enterprise SIEM corpus used in the paper is excluded for confidentiality reasons.

---

## Architecture

Three layers, each independently testable:

**Layer 1 (Ingestion):** All sources normalize to a shared `VulnerabilityEntity` tuple containing: unique ID, severity score (CVSS-normalized 0-10), affected component, affected version range, and free-text description. This schema is defined in `connectors/entity_schema.py`.

**Layer 2 (Fusion):** BiLSTM-CRF extracts entities from free text. Attention-based CNN classifies relationships. Graph Attention Network predicts missing links from graph topology. GPT-4 generates and validates new candidate links from vulnerability descriptions. All triples load into an OWL knowledge graph aligned with the MITRE ATT&CK data model.

**Layer 3 (Detection):** XGBoost ensemble receives a 47-dimensional feature vector per vulnerability entity. FGSM adversarial training mixes clean and perturbed samples at epsilon=0.01. Classified entities map to MITRE ATT&CK techniques via nearest-neighbor lookup in technique embedding space.

See `docs/architecture.md` for full technical detail.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{gentyala2026cyberfuse,
  title     = {CyberFuse-CI: Adversarially Resilient Vulnerability Detection Through
               Heterogeneous Multi-Source Data Fusion and LLM-Augmented Reasoning},
  author    = {Gentyala, Sunil and Mudusu, Sunil Kumar and Mannam, Praveen Kumar
               and Allam, Sathish and Prakash, Rakesh},
  booktitle = {Proceedings of the Sixth International Conference on Emerging
               Techniques in Computational Intelligence (ICETCI 2026)},
  year      = {2026},
  publisher = {IEEE},
  address   = {Hyderabad, India}
}
```

---

## Authors

| Name | Affiliation | Contact |
|---|---|---|
| Sunil Gentyala | HCL America Inc. (HCLTech), Dallas TX | sunil.gentyala@ieee.org |
| Sunil Kumar Mudusu |Church Mutual Insurance Company S.I, Austin, TX |sunil.mudusu@ieee.org|
| Praveen Kumar Mannam | Salesforce | -- |
| Sathish Allam | -- | -- |
| Rakesh Prakash | -- | -- |

---

## License

MIT License. See [LICENSE](LICENSE) for full terms.
