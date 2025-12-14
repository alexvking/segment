from pypdf import PdfReader

reader = PdfReader("Alex_King_Probabilistic_Word_Segmentation.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print(text)
