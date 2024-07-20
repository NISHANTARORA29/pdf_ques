import pdfplumber
from transformers import pipeline

def extract_text_from_pdf(file_path):
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.replace('\n', ' ').replace('\r', '')
    return text


qa_pipeline = pipeline("question-answering")

def answer_question(text, question):
    result = qa_pipeline(question=question, context=text)
    return result['answer']

def process_pdf_and_answer_question(pdf_path, question):
    text = extract_text_from_pdf(pdf_path)
    processed_text = preprocess_text(text)
    answer = answer_question(processed_text, question)
    return answer

pdf_path = 'path/to/your/pdf_file.pdf'
question = 'Your question here'
answer = process_pdf_and_answer_question(pdf_path, question)
print(answer)
