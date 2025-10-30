# Cipher Code - AI-Powered Code Analysis & Video Processing Platform

> A sophisticated web application combining AI-powered code analysis with three specialized modes, advanced video processing, and multi-agent intelligence systems.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/gradio-5.44.1+-orange.svg)](https://gradio.app/)
[![xAI Grok](https://img.shields.io/badge/xAI-Grok-purple.svg)](https://x.ai/)

## 🌟 Features

### 🎓 AI Code Analysis - Three Specialized Modes

#### LEARNING Mode
Your patient coding mentor that helps you become a better programmer:
- **Step-by-Step Teaching**: Breaks problems into small, digestible steps
- **Plain English Explanations**: Every line explained in simple terms
- **Official Documentation**: Cites official docs with URLs for every API/method
- **Test Cases Included**: Shows input-to-output examples for verification
- **Interactive Learning**: Asks "Does this make sense?" before moving on
- **No Assumptions**: Clarifies language, environment, and constraints first
- **Concept Highlighting**: Emphasizes one key concept per step

#### POLISH Mode
Production-grade code refinement for Fortune 500 standards:
- **Security Hardening**: Eliminates path traversal, injection, overflow, race conditions
- **Performance Optimization**: O(n log n) or better algorithms, minimal allocations
- **Clean Code Standards**: No magic numbers, TODOs, or redundancy
- **Type Safety**: Full type hints and validation
- **Reliability Focus**: Input validation, graceful degradation, resource cleanup
- **Standards Compliance**: Adheres to official coding standards (PEP 8 & PEP 20 for Python)
- **Line-by-Line Explanations**: Detailed justification for every change

#### HARDCORE Mode
10-agent swarm system for comprehensive code analysis:
- **API Doc Diver**: Live documentation lookup for all function calls
- **Architect**: Structural analysis & MVC pattern recommendations
- **Security Auditor**: Automated vulnerability detection
- **Performance Profiler**: Code optimization suggestions
- **Code Poet**: Quality scoring & type hint recommendations
- **Test Engineer**: Automated test case generation
- **Future Proofer**: Python 3.13+ compatibility checks
- **Feature Innovator**: New feature suggestions
- **Cost Optimizer**: API call reduction strategies
- **Synthesis Engine**: Merges all outputs into 9.9/10 quality code
- **Proof of Work**: Detailed logs and metrics from all 10 agents

### 🎥 Advanced Video Processing
- **Precision Video Overlay**: ECC-based stabilization and alignment
- **Multi-threaded Processing**: Optimized batch frame processing
- **Millisecond Precision**: HH:MM:SS.mmm format for timeline control
- **Adaptive Quality**: Resolution scaling (0.1-1.0x) and frame skip (1-10)
- **Frame Navigation**: Arrow keys for frame-by-frame control
- **Quick Timeline Setting**: Keyboard shortcuts (F/G keys) to set times
- **Large File Support**: Handles videos up to 10GB
- **Format Support**: MP4, MOV, AVI, MKV

### 📎 File Upload Support
- **Images**: JPG, PNG, GIF, WebP (up to 50MB)
- **Code Files**: Python, JavaScript, C/C++, Java, HTML, CSS, JSON, YAML, Markdown, and more (up to 10MB)
- **Direct Upload**: Attach button (📎) for quick file sharing
- **Vision Analysis**: Automatic image analysis using Grok vision model

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
pip install gradio openai opencv-python opencv-python-headless ffmpeg-python requests numpy
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

### Code Tab

#### Switching Modes
Click the mode button at the top to cycle through: **LEARNING** → **POLISH** → **HARDCORE**

#### LEARNING Mode Usage
1. Drop your code file or type your question
2. AI breaks down the solution step-by-step
3. Get explanations, test cases, and documentation links
4. Confirm understanding before moving to the next step

**Example Interaction:**
```
You: "How do I read a CSV file in Python?"

AI: 
Step 1: Import the csv module
[Shows code block]
[Explains each line]
[Links to official docs]
[Shows test case]
"Does this make sense? Ready for the next step?"
```

#### POLISH Mode Usage
1. Paste your code or upload a file
2. Get production-ready refactored code
3. Review line-by-line explanations of changes
4. Apply security, performance, and clean code improvements

**Example Output:**
```
Line 15: Changed os.getcwd() to UPLOAD_DIR (security: prevents path traversal)
Line 23: Added type hints (readability: int -> int)
Line 45: Replaced magic number with MAX_RETRIES constant (maintainability)
```

#### HARDCORE Mode Usage
1. Upload your codebase or paste code
2. 10 specialized agents analyze in parallel
3. Get comprehensive analysis with proof of work
4. Receive 9.9/10 quality code with unified diff

**Example Output:**
```python
Score: 9.9/10
Agents ran: 10
Duration: 3.42s

API Doc Diver: Found docs for 15 APIs
Security Auditor: Found 3 security risks
Performance Profiler: 2 optimization suggestions
[... detailed agent reports ...]
[... unified diff of changes ...]
```

### Video Tab

#### Basic Workflow
1. Upload **Base Video** and **Ghost Video**
2. Set **Opacity** (how transparent the ghost video appears)
3. Configure **Start Times** (HH:MM:SS.mmm format)
4. Optionally set **Duration** (how long to process)
5. Adjust **Frame Skip** (1-10, higher = faster but lower quality)
6. Set **Resolution Scale** (0.1-1.0, lower = faster processing)
7. Click **Process**

#### Keyboard Shortcuts
- **Arrow Keys**: Navigate frame-by-frame (click video first to focus)
- **F Key**: Set Base Start time from current video position
- **G Key**: Set Ghost Start time from current video position

#### Example Settings
**High Quality (slower):**
- Frame Skip: 1
- Resolution Scale: 1.0
- Opacity: 0.5

**Fast Preview (faster):**
- Frame Skip: 5
- Resolution Scale: 0.5
- Opacity: 0.5

## 🛠️ Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `XAI_API_KEY` | xAI API key (required) | None |
| `PORT` | Server port | 5000 |

### Video Processing Limits
| Setting | Range | Default |
|---------|-------|---------|
| Opacity | 0.1 - 1.0 | 0.5 |
| Frame Skip | 1 - 10 | 5 |
| Resolution Scale | 0.1 - 1.0 | 0.5 |
| Max File Size | - | 10GB |
| Max Duration | - | 24 hours |

### File Upload Limits
| File Type | Max Size | Supported Extensions |
|-----------|----------|---------------------|
| Images | 50MB | .jpg, .jpeg, .png, .gif, .webp |
| Code Files | 10MB | .py, .js, .cpp, .java, .html, .css, .json, .md, etc. |
| Videos | 10GB | .mp4, .mov, .avi, .mkv |

## 🏗️ Architecture

### Frontend
- Gradio-based web interface with custom CSS
- Light/dark theme toggle (half-circle button)
- Responsive mobile-first design
- JavaScript keyboard shortcuts for video control

### Backend
- Python with OpenCV for video processing
- xAI Grok API integration with streaming responses
- Multi-threaded agent execution (ThreadPoolExecutor)
- Secure temporary file handling
- ECC (Enhanced Correlation Coefficient) motion estimation

### AI Models
- `grok-code-fast-1`: Rapid text-based code analysis
- `grok-4-fast-reasoning`: Advanced vision and reasoning tasks

### Video Processing Pipeline
1. Input validation and file size checks
2. Video capture and FPS detection
3. Timeline positioning (start times + duration)
4. Frame extraction with skip optimization
5. ECC-based frame alignment
6. Alpha blending (weighted overlay)
7. Multi-threaded batch processing
8. H264 encoding and output

## 🔒 Security

- **Path Traversal Protection**: All file paths validated and sandboxed
- **File Size Limits**: 10GB for videos, 50MB for images, 10MB for code
- **Secure Temp Storage**: Automatic cleanup on exit
- **API Key Management**: Environment-based secrets
- **Input Sanitization**: Comprehensive validation for all user inputs
- **Error Isolation**: Prevents information leakage through error messages

## 📊 Performance

- **Multi-threading**: Up to 8 concurrent threads for processing
- **Batch Operations**: Reduces I/O overhead for video frames
- **Smart Caching**: Transformation matrix caching for alignment
- **Adaptive Algorithms**: ECC fallback for challenging content
- **Progress Tracking**: Real-time updates every 100 frames
- **Memory Efficient**: Batch processing prevents memory overflow

## 🎯 Use Cases

### For Learners
- Use **LEARNING mode** to understand new concepts step-by-step
- Upload code snippets and get patient explanations
- Learn best practices with official documentation links

### For Production Teams
- Use **POLISH mode** to prepare code for production
- Get security audits and performance optimizations
- Ensure compliance with coding standards

### For Code Quality
- Use **HARDCORE mode** for comprehensive analysis
- Get insights from 10 specialized agents
- Receive actionable improvement suggestions

### For Video Creators
- Overlay gameplay videos with different perspectives
- Create ghost comparisons for sports analysis
- Combine footage from multiple camera angles
- Generate transparent overlay effects

## 🤝 Contributing

This is a personal project, but suggestions and feedback are welcome!

## 📄 License

This project is provided as-is for educational and personal use.

## 🔗 Links

- [xAI Platform](https://x.ai/)
- [Gradio Documentation](https://gradio.app/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Python Documentation](https://docs.python.org/)

## 📮 Support

For issues or questions, please refer to the detailed technical documentation in `replit.md`.

## 🎯 Roadmap

- [ ] Add GIF export functionality
- [ ] Implement real-time video preview during processing
- [ ] Expand HARDCORE mode with additional specialized agents
- [ ] Add batch video processing for multiple files
- [ ] Implement video filters and color grading
- [ ] Add support for more video codecs
- [ ] Create API endpoint for programmatic access

## 🙏 Acknowledgments

Built with cutting-edge AI technology from xAI and the powerful Gradio framework for rapid web development.

---

**Built with ❤️ using Python, Gradio, and xAI Grok**
