FROM python:3.11-slim

WORKDIR /app

# system deps kept minimal; pdfplumber/faiss wheels need no compiler
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
