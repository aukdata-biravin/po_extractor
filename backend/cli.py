"""
Command-line interface for Purchase Order Extraction.

Usage:
    python cli.py extract path/to/pdf.pdf
    python cli.py extract-batch path/to/folder/
    python cli.py extract path/to/pdf.pdf --text-only
    python cli.py info path/to/pdf.pdf
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env so NVIDIA_API_KEY is available before importing config
load_dotenv(Path(__file__).parent / ".env")

from extractor import POExtractor  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract Purchase Order data from PDF documents"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file path (JSON). If not specified, outputs to stdout"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract PO from a single PDF")
    extract_parser.add_argument("pdf_file", type=Path, help="Path to PDF file")
    extract_parser.add_argument(
        "--text-only",
        action="store_true",
        help="Extract text only without AI parsing"
    )
    extract_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get info about a PDF file")
    info_parser.add_argument("pdf_file", type=Path, help="Path to PDF file")
    info_parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        extractor = POExtractor()
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        return 1
    
    try:
        if args.command == "extract":
            return handle_extract(extractor, args)
        elif args.command == "info":
            return handle_info(extractor, args)
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


def handle_extract(extractor: POExtractor, args) -> int:
    """Handle single file extraction."""
    pdf_file = Path(args.pdf_file)
    
    if not pdf_file.exists():
        logger.error(f"File not found: {pdf_file}")
        return 1
    
    logger.info(f"Extracting PO from: {pdf_file}")
    
    try:
        if args.text_only:
            text = extractor.extract_text_only(pdf_file)
            output = {"extracted_text": text}
        else:
            result = extractor.extract_from_pdf(pdf_file)
            output = result.model_dump()
        
        # Format output
        if args.pretty:
            json_str = json.dumps(output, indent=2)
        else:
            json_str = json.dumps(output)
        
        # Write to file or stdout
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                f.write(json_str)
            logger.info(f"Output written to: {args.output}")
        else:
            print(json_str)
        
        logger.info("Extraction completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return 1


def handle_info(extractor: POExtractor, args) -> int:
    """Handle PDF info retrieval."""
    pdf_file = Path(args.pdf_file)
    
    if not pdf_file.exists():
        logger.error(f"File not found: {pdf_file}")
        return 1
    
    try:
        info = extractor.get_pdf_info(pdf_file)
        
        if args.pretty:
            json_str = json.dumps(info, indent=2)
        else:
            json_str = json.dumps(info)
        
        print(json_str)
        return 0
        
    except Exception as e:
        logger.error(f"Failed to get PDF info: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
