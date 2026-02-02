import sys
try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf not installed")
    sys.exit(1)

try:
    reader = PdfReader("main.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    # Print a good chunk of text. The terminal output limit is 60KB.
    # A standard paper is around 20-30KB text.
    print(text) 
except Exception as e:
    print(f"Error reading PDF: {e}")
