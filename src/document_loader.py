import re
from pypdf import PdfReader
from pathlib import Path


# Function to load a PDF file
def load_pdf(file_path):

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:

            # Fix words split across PDF lines
            text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)

            # Normalize unnecessary whitespace
            text = re.sub(r"\s+", " ", text).strip()

        if text and text.strip():

            documents.append({
                "text": text,
                "source": Path(file_path).name,
                "page": page_number
            })

    return documents


# Function to load a TXT file
def load_txt(file_path):

    with open(file_path, "r", encoding="utf-8") as file:

        text = file.read()

    documents = []

    if text.strip():

        documents.append({
            "text": text,
            "source": Path(file_path).name,
            "page": None
        })

    return documents


# Function to identify and load documents
def load_document(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        return load_pdf(file_path)

    elif extension == ".txt":

        return load_txt(file_path)

    else:

        raise ValueError("Only PDF and TXT files are supported.")


# Main program
if __name__ == "__main__":

    file_path = "data/documents/Python Programming.pdf"

    documents = load_document(file_path)

    print("\nDocument Loading Results\n")

    for document in documents:

        print("Source:", document["source"])
        print("Page:", document["page"])
        print("Extracted Text:")
        print(document["text"])
        print("-" * 50)
