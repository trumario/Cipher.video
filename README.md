# Cipher - AI-Powered Code Analysis & Video Processing Platform

> A sophisticated web application combining AI-powered code analysis, advanced video processing, and multi-agent intelligence systems.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/gradio-5.44.1+-orange.svg)](https://gradio.app/)
[![xAI Grok](https://img.shields.io/badge/xAI-Grok-purple.svg)](https://x.ai/)

## 🌟 Features

### AI Code Analysis
- Intelligent code review using xAI's Grok models
- Multi-modal input (text and image URLs)
- Real-time streaming responses
- Conversation memory across sessions
- Security auditing and performance optimization

### Video Processing
- Precision video overlay with ECC stabilization
- Multi-threaded batch frame processing
- Custom timeline control with millisecond precision
- Adaptive quality settings (resolution scaling, frame skip)
- Supports files up to 10GB (MP4, MOV, AVI, MKV)

### 🔥 Hardcore Mode (10-Agent System)
Comprehensive code analysis through specialized agents:
- **API Doc Diver**: Live documentation lookup
- **Architect**: Structural analysis & MVC recommendations
- **Security Auditor**: Vulnerability detection
- **Performance Profiler**: Optimization suggestions
- **Code Poet**: Quality scoring & type hints
- **Test Engineer**: Automated test generation
- **Future Proofer**: Python 3.13+ compatibility
- **Feature Innovator**: New feature suggestions
- **Cost Optimizer**: API call reduction
- **Synthesis Engine**: Merges outputs into 9.9/10 code

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- FFmpeg installed
- xAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cipher
```

2. Install dependencies:
```bash
pip install gradio openai opencv-python opencv-python-headless ffmpeg-python requests
```

3. Set up environment variables:
```bash
export XAI_API_KEY="your_xai_api_key_here"
export PORT=5000  # Optional, defaults to 5000
```

4. Run the application:
```bash
python main.py
```

5. Open your browser to `http://localhost:5000`

## 📁 Project Structure

```
cipher/
├── main.py                 # Main Gradio application
├── hardcore.py             # 10-agent hardcore analysis system
├── static/                # Static assets
├── attached_assets/       # User-uploaded assets
├── README.md             # This file
└── replit.md             # Detailed technical documentation
```

## 🎮 Usage

### Code Analysis Tab
1. Enter code or paste an image URL
2. Get real-time AI-powered analysis
3. Receive security audits, optimization suggestions, and refactoring recommendations

### Video Processing Tab
1. Upload base and ghost videos
2. Adjust opacity, start times, and duration
3. Configure frame skip and resolution scale
4. Process and download the overlay result

### Hardcore Mode
```python
from hardcore import run_hardcore_mode

# Analyze code with 10 agents
final_code, diff, proof = run_hardcore_mode(original_code, intensity=9.9)

print(f"Score: {proof['score']}")
print(f"Agents ran: {proof['agents_ran']}")
print(f"Duration: {proof['duration']}")
print("\nDiff:\n", diff)
```

## 🛠️ Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `XAI_API_KEY` | xAI API key (required) | None |
| `PORT` | Server port | 5000 |

### Video Processing Settings
| Setting | Range | Default |
|---------|-------|---------|
| Opacity | 0.1 - 1.0 | 0.5 |
| Frame Skip | 1+ | 1 |
| Resolution Scale | 0.1 - 1.0 | 1.0 |
| Max File Size | - | 10GB |

## 🏗️ Architecture

### Frontend
- Gradio-based web interface
- Custom CSS with light/dark theme
- Responsive mobile-first design
- JavaScript keyboard shortcuts

### Backend
- Python with OpenCV for video processing
- xAI Grok API integration
- Multi-threaded agent execution
- ThreadPoolExecutor for parallel processing

### AI Models
- `grok-code-fast-1`: Rapid code analysis
- `grok-4-0709`: Image and visual analysis

## 🔒 Security

- Input validation and sanitization
- File size limits (10GB max)
- Secure file path handling
- Environment-based API key management
- Error isolation to prevent information leakage

## 📊 Performance

- Multi-threaded video processing
- Batch frame operations
- Smart transformation matrix caching
- ECC fallback algorithms
- Parallel agent execution (up to 10 threads)

## 🤝 Contributing

This is a personal project, but suggestions and feedback are welcome!

## 📄 License

This project is provided as-is for educational and personal use.

## 🔗 Links

- [xAI Platform](https://x.ai/)
- [Gradio Documentation](https://gradio.app/)
- [OpenCV Documentation](https://docs.opencv.org/)

## 📮 Support

For issues or questions, please refer to the detailed technical documentation in `replit.md`.

## 🎯 Roadmap

- [ ] Add GIF export functionality
- [ ] Implement real-time video preview
- [ ] Add more agent types to Hardcore mode
- [ ] Add batch video processing
- [ ] Implement video filters and effects

---

**Built with ❤️ using Python, Gradio, and xAI Grok**
