# Doomscroll & Ragebait Classifier

A synthetic supervised fine-tuning (SFT) dataset generator and classifier for social media content, detecting doomscroll hooks, brainrot, and ragebait using the Google Gemini API.

## Features

- **Synthetic SFT Dataset Generator**: Generates high-quality labeled examples (`input_text` -> `label` + `reasoning`) with structured JSON schema outputs.
- **Labels**:
  - `high-dopamine ragebait`
  - `brainrot/low-attention`
  - `educational`
  - `neutral`
- **Rate-Limited Pipeline**: Token-bucket rate limiter with exponential backoff designed for Google AI Studio free tier limits.
- **Agent Integration**: Agent runner powered by `google.antigravity`.

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd doomscroll_classifier
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Copy `.env.example` to `.env` and set your Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Add your API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Usage

### Generate Synthetic Dataset
```bash
python generate_dataset.py --num-batches 10
```

### Run the Agent
```bash
python agent.py
```
