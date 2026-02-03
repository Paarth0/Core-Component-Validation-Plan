
# Core-Component-Validation-Plan
# PKA Validation Core

Laptop-based validation framework for the iPhone 17 Pro Personal Knowledge Assistant (PKA v2.0) architecture.

##  Project Overview

This project validates that the core document processing pipeline from the PKA specification can run on standard laptop hardware using open-source tools.
The implementation proves that the iPhone 17 Pro PKA architecture can be validated on standard laptop hardware using open-source tools.


## Project Structure
pka-validation/
├── README.md                      # This file
├── DECISIONS.md                   # Architecture Decision Records
├── requirements.txt               # Python dependencies
├── pytest.ini                     # Pytest configuration
├── src/
│   ├── main.py                    # CLI entry point
│   ├── ingestion_coordinator.py   # Task 1.1 - Pipeline orchestration
│   ├── extraction_worker.py       # Task 1.2 - Document parsing
│   ├── text_splitter.py           # Task 2.1 - Semantic chunking
│   ├── embedding_worker.py        # Task 2.2 - Vector embeddings
│   └── indexing_worker.py         # Vector storage (SQLite)
├── tests/
│   ├── test_golden_output.py      # Golden output tests
│   └── test_acceptance_criteria.py # AC validation tests
├── scripts/
│   └── create_test_pdf.py         # Test PDF generator
├── data/                          # Input documents
├── models/                        # Cached ML models
└── output/                        # Processing results



## Requirements Compliance

| Requirement | Description | Status |
|-------------|-------------|--------|
| CFR-1 | Parse PDF, DOCX, XLSX, XML | ✅ Passed |
| CFR-2 | 512-token chunks, 10% overlap | ✅ Passed |
| CFR-3 | Float16 vector storage | ✅ Passed |
| CNFR-1 | Offline-first processing | ✅ Passed |
| CNFR-2 | CPU-only multi-core | ✅ Passed |
| Task 1.1 | IngestionCoordinator (asyncio) | ✅ Passed |
| Task 1.2 | XLSX <500MB RAM | ✅ Passed (1.6MB used) |
| Task 2.1 | RecursiveCharacterTextSplitter | ✅ Passed |
| Task 2.2 | Embedding <50ms latency | ✅ Passed (1.23ms) |


### Prerequisites
- Python 3.9+
- macOS / Linux / Windows


# Core Component Validation Plan
This repository contains the Core Component Validation Plan project.  
Follow the steps below to set up and run the project on any computer.


## Installation
# 1. Clone the repository
```bash
git clone https://github.com/Paarth0/Core-Component-Validation-Plan.git
cd Core-Component-Validation-Plan

# 2. Create a virtual environment
python3 -m venv venv
(Mac) source venv/bin/activate | (WIndows) venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


### Running the Pipeline
# Create test PDF
python scripts/create_test_pdf.py

# Ingest a document
python src/main.py ingest --input ./data/sample_5page.pdf --verbose

# Search the database
python src/main.py search --query "validation methodology" --top-k 3

# View statistics
python src/main.py stats


### Running Tests
1. Testing Search Functionality
   Run Search Commands

cd /Users/a]--/pka-validation
source venv/bin/activate

# Search 1: Find content about introduction
python src/main.py search --query "introduction validation core" --top-k 3

# Search 2: Find content about methodology
python src/main.py search --query "document extraction chunking" --top-k 3

# Search 3: Find content about results
python src/main.py search --query "performance metrics latency" --top-k 3

# Search 4: Find content about conclusion
python src/main.py search --query "production implementation iPhone" --top-k 3

# Run all tests
PYTHONPATH=./src pytest tests/ -v -s

# Run specific test file
PYTHONPATH=./src pytest tests/test_golden_output.py -v -s
