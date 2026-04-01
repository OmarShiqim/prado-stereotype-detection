# Prado Stereotype Detection

Computational detection of social stereotypes in artworks from the 
Prado Museum using Multimodal Large Language Models and SADCAT 
dictionary-based scoring grounded in the Stereotype Content Model (SCM).

## Project Overview

This project develops a reproducible end-to-end pipeline that:
1. Generates rich textual descriptions of artwork images using LLaVA
2. Scores those descriptions using the SADCAT dictionary system to 
   produce warmth and competence scores
3. Applies statistical analysis to detect stereotype patterns across 
   iconographic categories

The pipeline was applied to 6,266 paintings from the Prado Museum 
collection, producing warmth and competence scores for over 99.9% 
of the dataset.

## Repository Structure

| File | Description |
|---|---|
| `00_data_audit.ipynb` | Database exploration, image coverage analysis, and data quality validation |
| `01_blip2_pipeline.ipynb` | BLIP-2 model evaluation and smoke test |
| `02_llava_pipeline.ipynb` | Full LLaVA description generation pipeline |
| `03_deepseek_pipeline.ipynb` | DeepSeek-VL pipeline setup and configuration |
| `04_llava_validation.ipynb` | Post-generation quality checks on LLaVA descriptions |
| `05_analysis.ipynb` | Hypothesis testing, temporal analysis, and context effects analysis |
| `06_museum_analysis.ipynb` | Museum description analysis and LLaVA vs museum comparison |
| `SADCAT_Scoring_llava_Omar.nb.html` | R notebook implementing SADCAT scoring pipeline using AGRUPA library |
| `run_llava.py` | Production script for running LLaVA pipeline as background process |
| `test_sadcat_prompt.py` | Script for testing SADCAT-guided prompt on sample artworks |

## Technical Requirements

- Python 3.12.3
- CUDA 12.8
- R 4.3 with AGRUPA and SADCAT packages
- GPU with at least 16GB VRAM recommended

Install Python dependencies:
```
pip install -r requirements.txt
```

## Data

The data used in this project was obtained from the Prado Museum's 
digital collection and is not included in this repository. The pipeline 
requires access to the project SQLite database (`agrupa.sqlite`) 
containing artwork metadata, image file paths, and museum descriptions.

## Citation

If you use this code or methodology in your work, please cite:

Shiqim, O. (2026). Computational Detection of Social Stereotypes in 
Artworks from the Prado Museum. IE University Capstone Thesis.

## Acknowledgements

This project was developed as part of the BDBA Capstone at IE University 
under the supervision of Alejandro Martínez-Mingo. 
