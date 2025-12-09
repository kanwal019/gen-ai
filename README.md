# Gen-AI

A comprehensive learning project exploring generative AI applications using LangChain, Ollama, and machine learning techniques.

## Project Overview

This repository contains multiple AI-powered applications demonstrating:
- **Product Description Generation** - Automated generation of detailed product descriptions using LLMs
- **Anomaly Detection** - Fine-tuned models for detecting faulty sensor readings
- **Custom AI Models** - Integration with Ollama for local LLM inference

## Key Features

- 🤖 **LLM Integration** - Uses Ollama with LangChain for conversational AI
- 📝 **Product Description Generator** - AI-powered tool for creating compelling product descriptions
- 🔍 **Anomaly Detection** - Machine learning pipeline for detecting faulty data in sensor readings
- 🎯 **Custom Model Configuration** - Modelfile support for fine-tuned AI personalities
- 📊 **Data Processing** - Pandas-based data preprocessing and feature engineering

## Technologies Used

- **LLMs**: Ollama (Llama 3.1, Mistral)
- **Framework**: LangChain & LangChain-Ollama
- **Data Science**: Pandas, Scikit-learn
- **Python**: 3.10+

## Project Structure

```
gen-ai/
├── main.py                 # Product description generator with conversational interface
├── load.py                 # Anomaly detection model training and evaluation
├── Modelfile               # Custom Ollama model configuration (Jarvis personality)
├── requirements.txt        # Python dependencies
└── chatbot/                # Virtual environment directory
```

## Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gen-ai
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv chatbot
   chatbot\Scripts\Activate.ps1  # On Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Ollama is running**
   ```bash
   ollama serve
   ```
   
   Pull required models:
   ```bash
   ollama pull llama3.1
   ollama pull mistral
   ```

## Usage

### Product Description Generator

Run the interactive product description generator:

```bash
python main.py
```

**Example Input:**
```
Create a description for product code WFO with following attributes - 
Name - 'Optimus - 20 oz. Double Wall Stainless Tumbler with Ceramic Interior'
Material(Inner & Outer) - Recycled 304 Stainless Steel, Lid Clear AS
Weight - 283.20 g
Dimensions - 19.2 cm height with lid x 8.6 cm diameter
```

The generator will create detailed, creative product descriptions with key features, benefits, and unique selling points.

### Anomaly Detection Model

Train the anomaly detection model:

```bash
python load.py
```

**What it does:**
- Loads normal and faulty sensor data from CSV files
- Preprocesses and normalizes sensor readings
- Extracts temporal features (hour, day, month, year)
- Fine-tunes an Ollama model on the labeled dataset
- Evaluates performance on test data
- Saves the trained model as 'operate_mark1'

**Required CSV files:**
- `normal_data.csv` - Normal sensor readings with timestamp
- `faulty_data.csv` - Faulty sensor readings with timestamp

### Custom Ollama Model

Build and use the custom Jarvis-personality model:

```bash
ollama create jarvis -f Modelfile
ollama run jarvis
```

## Dependencies

- **langchain** - LLM orchestration framework
- **langchain_ollama** - Ollama integration for LangChain
- **ollama** - Python client for Ollama API
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning preprocessing and model selection

See `requirements.txt` for complete dependency list with versions.

## Contributing

This is a learning project. Feel free to fork, modify, and experiment with different models and approaches.

## License

See LICENSE file for details.

## Notes

- The project uses local Ollama models, so no API keys are required
- Ensure Ollama service is running before executing main.py
- For production use, consider implementing proper error handling and logging
- The product description generator maintains conversation context for multi-turn interactions

