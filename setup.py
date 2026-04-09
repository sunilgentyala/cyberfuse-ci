"""
setup.py for cyberfuse-ci
"""
from setuptools import setup, find_packages

setup(
    name="cyberfuse-ci",
    version="1.0.0",
    description=(
        "CyberFuse-CI: Adversarially Resilient Vulnerability Detection "
        "Through Heterogeneous Multi-Source Data Fusion and LLM-Augmented Reasoning"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Sunil Gentyala, Sunil Kumar Mudusu, Praveen Kumar Mannam, Sathish Allam, Rakesh Prakash",
    author_email="sunil.gentyala@ieee.org",
    url="https://github.com/sunilgentyala/cyberfuse-ci",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "torchcrf>=1.1.0",
        "transformers>=4.40.0",
        "xgboost>=2.0.3",
        "scikit-learn>=1.4.0",
        "imbalanced-learn>=0.12.0",
        "requests>=2.31.0",
        "stix2>=3.0.1",
        "taxii2-client>=2.3.0",
        "rdflib>=7.0.0",
        "owlready2>=0.46",
        "openai>=1.30.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "jsonlines>=4.0.0",
        "tqdm>=4.66.0",
        "mitreattack-python>=3.0.2",
    ],
    extras_require={
        "faiss": ["faiss-cpu>=1.8.0", "sentence-transformers>=3.0.0"],
        "dev":   ["pytest>=8.0.0", "pytest-cov>=5.0.0", "responses>=0.25.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "cybersecurity", "vulnerability-detection", "adversarial-machine-learning",
        "LLM", "knowledge-graph", "MITRE-ATTACK", "data-fusion", "computational-intelligence"
    ],
)
