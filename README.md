# prompt_injection_detector
A Python-based security tool for detecting Prompt Injection Attacks in LLM applications. This detector combines regex-based rules, indirect document scanning, and semantic similarity with embeddings to identify malicious or harmful instructions that could compromise AI safety.

1. System Requirements
Python 3.9+ (Python installed hona chahiye)
Pip (package installer)

pip install streamlit google-generativeai python-dotenv sentence-transformers


ap Google Gemini (Generative AI) ka use karna chahte ho semantic detection ke liye →
.env file banaiye aur likhiye:

GOOGLE_API_KEY=your_api_key_here

streamlit run app.py
