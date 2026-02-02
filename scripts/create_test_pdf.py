#!/usr/bin/env python3
"""
Create a test PDF for golden output testing.

Usage:
    python scripts/create_test_pdf.py

Creates: data/sample_5page.pdf
"""

from pathlib import Path
import sys


def create_test_pdf():
    """Create a 5-page test PDF with known content."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
    except ImportError:
        print("ERROR: reportlab not installed.")
        print("Install with: pip install reportlab")
        sys.exit(1)
    
    # Get project root (parent of scripts folder)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    output_path = data_dir / "sample_5page.pdf"
    
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter
    
    # Page content - 5 pages with structured content
    pages = [
        {
            "title": "Introduction",
            "content": [
                "This document serves as a test case for the PKA Validation Core.",
                "It contains structured content across multiple pages.",
                "The introduction provides context for the entire document.",
                "",
                "Key topics covered in this document:",
                "- Document processing pipeline validation",
                "- Semantic chunking verification",
                "- Vector embedding quality checks",
                "- End-to-end integration testing",
                "",
                "This content should be properly extracted and chunked.",
                "Each section contains unique keywords for validation.",
            ]
        },
        {
            "title": "Background",
            "content": [
                "The Personal Knowledge Assistant (PKA) is designed to process",
                "documents locally on device without network connectivity.",
                "",
                "Core requirements include:",
                "1. Multimodal ingestion of PDF, DOCX, XLSX, and XML files",
                "2. Semantic text chunking with 512-token windows",
                "3. Vector embeddings stored in Float16 format",
                "4. Completely offline operation",
                "",
                "This validation framework proves these requirements on laptops.",
                "The background section explains the system architecture.",
            ]
        },
        {
            "title": "Methodology",
            "content": [
                "The validation methodology consists of several phases:",
                "",
                "Phase 1: Document Extraction",
                "- Parse input documents using format-specific handlers",
                "- Preserve document structure and layout",
                "- Handle tables and special formatting",
                "",
                "Phase 2: Text Chunking",
                "- Apply recursive character text splitting",
                "- Maintain semantic boundaries",
                "- Ensure proper overlap between chunks",
                "",
                "Phase 3: Embedding Generation",
                "- Generate vector representations using ONNX models",
                "- Verify latency meets performance targets",
            ]
        },
        {
            "title": "Results",
            "content": [
                "Testing has demonstrated the following results:",
                "",
                "Performance Metrics:",
                "- Average embedding latency: <50ms per chunk",
                "- Memory usage for 10MB XLSX: <500MB",
                "- Processing speed: 10+ pages per second",
                "",
                "Quality Metrics:",
                "- No content loss during chunking",
                "- Proper overlap between consecutive chunks",
                "- Accurate vector similarity search",
                "",
                "All acceptance criteria have been met successfully.",
            ]
        },
        {
            "title": "Conclusion",
            "content": [
                "This validation framework successfully demonstrates that",
                "the PKA document processing pipeline can run on standard",
                "laptop hardware using open-source components.",
                "",
                "Key achievements:",
                "- Cross-platform compatibility verified",
                "- Performance targets achieved on CPU",
                "- Offline operation confirmed",
                "",
                "The architecture is ready for production implementation",
                "on the target iPhone 17 Pro hardware.",
                "",
                "End of document.",
            ]
        }
    ]
    
    for i, page in enumerate(pages, 1):
        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(1 * inch, height - 1 * inch, f"Chapter {i}: {page['title']}")
        
        # Underline
        c.line(1 * inch, height - 1.1 * inch, 6 * inch, height - 1.1 * inch)
        
        # Content
        c.setFont("Helvetica", 11)
        y = height - 1.5 * inch
        
        for line in page["content"]:
            if y < 1 * inch:
                break
            c.drawString(1 * inch, y, line)
            y -= 0.25 * inch
        
        # Page number
        c.setFont("Helvetica", 10)
        c.drawString(width / 2 - 20, 0.5 * inch, f"Page {i} of {len(pages)}")
        
        # New page (except for last)
        if i < len(pages):
            c.showPage()
    
    c.save()
    
    print(f"✓ Created test PDF: {output_path}")
    print(f"  Pages: {len(pages)}")
    print(f"  Size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\nYou can now run:")
    print(f"  python src/main.py ingest --input ./data/sample_5page.pdf --verbose")


if __name__ == "__main__":
    create_test_pdf()