# Document Chatbot with Ollama

A simple Flask chatbot that answers questions from uploaded documents using local LLMs via Ollama.

## Features

- Upload PDF, Word, Excel, CSV files
- Ask questions about document content
- Choose from available Ollama models
- Clean chat interface

## Setup

### 1. Install Ollama
Download from [ollama.ai](https://ollama.ai)

### 2. Pull a model
```bash
ollama pull llama3.2
```

### 3. Start Ollama
```bash
ollama serve
```

### 4. Install Python dependencies
```bash
cd DocumentChatbot
pip install -r requirements.txt
```

### 5. Run the app
```bash
python app.py
```

### 6. Open browser
Go to `http://localhost:5000`

## Usage

1. Select a model from dropdown
2. Upload a document
3. Ask questions in the chat

## Project Structure

```
DocumentChatbot/
├── app.py              # Flask backend (single file)
├── requirements.txt    # Dependencies
├── templates/
│   └── index.html      # Frontend
└── uploads/            # Uploaded files
```
