# Personal Healthcare Assistant

A multimodal AI-powered personal healthcare assistant built with Google's Gemma-3N model, designed to provide reliable and accessible healthcare advice using efficient small language models that can run on small-scale hardware.

## Overview

This project addresses the challenge of providing quality healthcare guidance in resource-constrained environments where access to large-scale models and multiple GPUs isn't available. By leveraging Google's small Gemma-3N model and fine-tuned medical variants, this application offers both text and multimodal healthcare assistance.

### Key Features

- **Multimodal Input Support**
  - Text queries via chat interface
  - Image upload and analysis (JPG, JPEG, PNG, WEBP)
  - Voice input with automatic speech recognition (ASR)
  - Any combination of these

- **Dual Interface Options**
  - **Web Interface** (`home.py`): Full-featured Streamlit GUI with file uploads, voice input, and chat history
  - **CLI Interface** (`chatbot.py`): Lightweight command-line tool for terminal-based interactions

- **Advanced Capabilities**
  - Conversational memory within each chat session
  - Fine-tuned medical models for healthcare-specific queries
  - Real-time token usage and latency tracking
  - Configurable model parameters (temperature, max tokens, top-p)

- **Specialized Models**
  - Base Gemma-3N for general multimodal tasks
  - Fine-tuned Gemma-3N for text-only queries
  - MedGemma-27B for medical image analysis

## Project Structure

```
kaggle-gemma-3n-challenge/
├── home.py             # Streamlit web interface
├── chatbot.py          # Command-line interface
├── utility.py          # Helper functions (ASR, file management)
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
├── input-files/        # Directory for uploaded files
├── db/                 # ChromaDB storage
└── README.md           # This file
```

## Installation

### Prerequisites

- Python 3.11+
- Internet access for vLLM-compatible API endpoints (or configure your own)

### Setup

1. **Clone the repository**
```
git clone https://github.com/ashen-321/kaggle-gemma-3n-challenge.git
cd kaggle-gemma-3n-challenge
```

2. **Install dependencies**
```
pip install -r requirements.txt
```

## Usage

### Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run home.py
```

The application will open in your browser with:
- **Sidebar**: Upload images, configure model parameters, upload voice queries
- **Main Chat**: Interact with the assistant
- **Settings**: Adjust various response generation parameters (temperature, top-p, max tokens, etc.)

#### Features:
- Upload multiple images via the sidebar
- Toggle microphone to record voice queries
- Clear chat history to start fresh conversations
- View real-time token usage and response latency

### Command-Line Interface

Run the console-based chatbot:

```bash
python chatbot.py
```

#### CLI Commands:
- **Regular query**: Type your question and press Enter
- **quit**: Exit the program
- **wipe**: Clear all message history
- **File inclusion**: Place files in `input-files/` directory
- **Ignore files**: Prefix filename with `#` (e.g., `#image.jpg`)

#### CLI File Support:
```
# Add files to input-files directory
cp /path/to/image.jpg input-files/

# Query will automatically include the file
Query: What is in this image?

# Ignore specific files by prefixing with #
mv input-files/image.jpg input-files/#image.jpg
```

## Configuration

### Model Endpoints

The project uses OpenAI-compatible API endpoints configured in the code:

| Model | Endpoint | Purpose |
|-------|----------|---------|
| `google/gemma-3n-E4B-it` | `http://video.cavatar.info:8083/v1` | Base multimodal model |
| `alfredcs/gemma-3N-finetune` | `http://video.cavatar.info:8087/v1` | Fine-tuned text model |
| `alfredcs/torchrun-medgemma-27b-grpo-merged` | `http://mcp1.cavatar.info:8081/v1` | Medical image analysis |

To use your own endpoints, modify the `base_url` parameters in:
- **home.py**: Line 18
- **chatbot.py**: Lines 29, 34
- **utility.py**: Line 60

### Model Parameters (Web Interface)

Adjust these parameters in the sidebar:

- **Temperature** (0.0-1.0): Controls randomness
  - Lower = more focused and deterministic
  - Higher = more creative and varied

- **Max Tokens** (0-4096): Maximum response length

- **Top-p** (0.1-1.0): Nucleus sampling threshold
  - Lower = more conservative word choices
  - Higher = more diverse vocabulary

## Dependencies

```
streamlit              # Web interface framework
pillow                 # Image processing
numpy                  # Numerical operations
streamlit_pdf_viewer   # PDF rendering (optional)
openai                 # API client
requests               # HTTP requests for ASR
```

## API Compatibility

This project uses the OpenAI Python SDK with custom base URLs to connect to vLLM-hosted Gemma models. Any OpenAI-compatible inference server can be used by updating the endpoint URLs.

### Using Local vLLM

```bash
# Start vLLM server
vllm serve google/gemma-3n-E4B-it --port 8083

# Update base_url in code
base_url="http://localhost:8083/v1"
```

## Performance Metrics

The web interface displays real-time metrics:
- **Completion Tokens**: Tokens in model response
- **Prompt Tokens**: Tokens in user input
- **Total Tokens**: Sum of prompt + completion
- **Latency**: Response time in milliseconds

## Troubleshooting

### Common Issues

**1. API connection errors**
- Verify endpoint URLs are accessible
- Check internet connectivity
- Ensure API services are running

**2. Voice input not working**
- Grant microphone permissions in browser
- Check ASR service availability at `http://video.cavatar.info:8082` (or your own ASR address)

**3. File upload issues**
- Supported formats: JPG, JPEG, PNG, WEBP (images), MP3, WAV (audio)
- Ensure files are not corrupted
- Check file size limits

## Future Enhancements

- [ ] Local model deployment guide
- [ ] Mobile-responsive interface
- [ ] Export chat history
- [ ] Custom model fine-tuning instructions
- [ ] Docker containerization
- [ ] Authentication and user management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google for the Gemma-3N model series
- Kaggle and Google for hosting the challenge
- vLLM team for the inference server
- Streamlit for the web framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback about this project, please open an issue on GitHub.
