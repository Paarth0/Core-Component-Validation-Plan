
---

### `DECISIONS.md`

```markdown
# Architectural Decision Records

This document captures key architectural decisions made during the implementation of the PKA Validation Core.

---

## ADR-001: ONNX Runtime over Gemma 3n

**Status:** Accepted

**Context:**
The original iPhone PKA specification uses Gemma 3n (~12GB) for embeddings, optimized for the A19 Pro Neural Engine.

**Decision:**
Use `all-MiniLM-L6-v2` exported to ONNX format (~80MB).

**Rationale:**
1. Gemma 3n requires 12GB+ RAM, unsuitable for laptop validation
2. ONNX Runtime provides efficient CPU inference
3. MiniLM produces quality 384-dimensional embeddings
4. Cross-platform compatibility (macOS, Linux, Windows)
5. Validates embedding pipeline without hardware dependencies

**Consequences:**
- Embedding quality may differ from production Gemma model
- Acceptable for validation; production uses target hardware
- Meets AC: <50ms latency per chunk

**Traceability:** 
- Spec Section 3: "Interim Core Substitute: Option C: Tiny Local Model (ONNX Runtime)"

---

## ADR-002: pdfplumber over Apple PDFKit

**Status:** Accepted

**Context:**
Apple PDFKit is iOS/macOS only and cannot run on Linux validation environment.

**Decision:**
Use `pdfplumber` Python library for PDF text extraction.

**Rationale:**
1. Pure Python, cross-platform compatibility
2. Layout-aware extraction (XY-Cut equivalent)
3. Table extraction capability
4. Active maintenance and documentation
5. No system dependencies

**Consequences:**
- Layout fidelity may differ slightly from PDFKit
- Performance may be slower than native implementation
- Acceptable for validation purposes

**Traceability:**
- Spec Section 8, Conflict: "Apple-only PDFKit"
- Fix: "Use pdfplumber for XY-Cut equivalent"

---

## ADR-003: Skip iWork .iwa Format

**Status:** Accepted

**Context:**
iWork files (Pages, Numbers, Keynote) use proprietary .iwa binary format.

**Decision:**
Do not implement native .iwa parsing. Use "Strategy B" - require PDF export.

**Rationale:**
1. .iwa format is undocumented and proprietary
2. Reverse engineering is time-consuming and fragile
3. Users can easily export to PDF from iWork apps
4. Focus resources on core validation

**Consequences:**
- Cannot directly ingest Keynote/Pages/Numbers files
- Users must export to PDF first
- Clear error message provided for unsupported formats

**Traceability:**
- Spec Section 8, Conflict: "iWork .iwa format"
- Fix: "Skip native parse; use Strategy B (PDF fallback)"

---

## ADR-004: Relaxed Performance Target

**Status:** Accepted

**Context:**
Original specification targets 100 pages/second using A19 Pro P-cores.

**Decision:**
Relax validation target to 10 pages/second (10x reduction).

**Rationale:**
1. Standard x86/ARM laptop CPUs cannot match A19 Pro performance
2. 10x relaxation allows meaningful validation on commodity hardware
3. Focus is on correctness and architecture, not performance parity
4. Performance benchmarking requires actual target hardware

**Consequences:**
- Validation proves algorithmic correctness, not production speed
- Performance testing deferred to device-specific validation
- Clear documentation of performance expectations

**Traceability:**
- Spec Section 8, Conflict: "100 pages/sec target"
- Fix: "Relax to 10 pages/sec for validation"

---

## ADR-005: Standard SQLite without sqlite-vec

**Status:** Accepted

**Context:**
sqlite-vec requires C extension compilation which varies by platform.

**Decision:**
Use standard SQLite with manual cosine similarity search. sqlite-vec is optional.

**Rationale:**
1. Simplifies local development and testing setup
2. Brute-force search acceptable for validation dataset sizes
3. sqlite-vec can be added later for production deployment
4. Reduces platform-specific build complexity

**Consequences:**
- Vector search is O(n) instead of approximate nearest neighbor
- Acceptable for validation with <100k vectors
- Production deployment can add sqlite-vec for performance

**Traceability:**
- Supports CNFR-2: CPU-only requirement
- R-3 (Low): SQLite Locking handled via WAL mode

---

## ADR-006: asyncio for Actor Pattern

**Status:** Accepted

**Context:**
Swift Actors provide isolated state and async execution. Need equivalent in Python.

**Decision:**
Use Python `asyncio` with `async/await` and `asyncio.to_thread()` for CPU-bound work.

**Rationale:**
1. asyncio provides native coroutine support
2. `asyncio.Lock` provides actor-like state isolation
3. `asyncio.to_thread()` offloads CPU work without blocking
4. Familiar pattern for Python developers
5. Good integration with I/O operations

**Consequences:**
- Not true actor isolation (Python has GIL)
- Sufficient for validation purposes
- Multi-threading via thread pool for CPU parallelism

**Traceability:**
- Task 1.1: "Build IngestionCoordinator using Python asyncio to mimic Swift Actors"
- CNFR-2: "Ingestion must utilize multi-core processing"

---

## ADR-007: tiktoken for Token Counting

**Status:** Accepted

**Context:**
CFR-2 specifies 512-token chunks. Need accurate token counting.

**Decision:**
Use OpenAI's `tiktoken` library with `cl100k_base` encoding.

**Rationale:**
1. Industry-standard tokenizer
2. Fast and accurate token counting
3. Compatible with most modern LLMs
4. Well-maintained and documented

**Consequences:**
- Token counts may differ slightly from target model's tokenizer
- Acceptable variance for chunking purposes
- Could be swapped for model-specific tokenizer if needed

**Traceability:**
- CFR-2: "512-token windows and 10% overlap"