import logging
from pathlib import Path

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

# Configuration with optimal settings for M3 Max
CONFIG = {
    "output_format": "markdown",
    "use_llm": False,
    "force_ocr": False,
}

def convert_pdf_to_text(pdf_path: str) -> str:
    """
    Converts a PDF file to plain text using the Marker library.
    
    Args:
        pdf_path (str): The path to the PDF file.
    
    Returns:
        str: The converted text.
    """
    try:
        logging.info(f"Converting {pdf_path} to text using Marker...")
        
        # Create converter instance with built-in parallelization
        config_parser = ConfigParser(CONFIG)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
        )
        
        rendered = converter(pdf_path)
        text = rendered.markdown

        output_path = Path(pdf_path).with_suffix('.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        logging.info(f"Conversion complete. Saved to {output_path}")
        return text
    except Exception as e:
        logging.error(f"Error converting {pdf_path}: {e}")
        raise

def process_pdf_folder(input_folder: str) -> None:
    """
    Process PDFs using Marker's built-in parallelization.
    
    Args:
        input_folder (str): Path to folder containing PDFs
    """
    input_path = Path(input_folder)
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        logging.warning(f"No PDF files found in {input_folder}")
        return

    logging.info(f"Found {len(pdf_files)} PDF files. Processing...")
    
    # Process files one at a time, letting Marker handle parallelization
    for pdf_file in pdf_files:
        try:
            convert_pdf_to_text(str(pdf_file))
        except Exception as e:
            logging.error(f"Failed to process {pdf_file}: {e}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    input_folder = "PDFS"
    process_pdf_folder(input_folder)
