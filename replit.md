
# Cipher - AI-Powered Code Analysis & Video Processing Platform

A sophisticated web application that combines AI-powered code analysis with advanced video processing capabilities. Built with Python and Gradio, featuring integration with xAI's Grok API for intelligent code review and a powerful video overlay system for precision video editing.

## Core Features

### 🤖 AI Code Analysis & Chat
- **Intelligent Code Review**: Advanced AI-powered code analysis using xAI's Grok models
- **Multi-modal Input**: Support for both text-based code queries and image analysis (URLs)
- **Streaming Responses**: Real-time AI responses with automatic continuation for long outputs
- **Conversation Memory**: Maintains chat history for contextual conversations across sessions
- **Security-First Analysis**: Built-in security auditing and vulnerability detection
- **Performance Optimization**: Automated code refactoring suggestions for efficiency improvements
- **Cross-Language Support**: Expert analysis across multiple programming languages

### 🎥 Advanced Video Processing
- **Precision Video Overlay**: Pixel-perfect alignment of two video streams with ECC (Enhanced Correlation Coefficient) stabilization
- **Multi-threaded Processing**: Optimized performance using ThreadPoolExecutor for batch frame processing
- **Flexible Timeline Control**: Support for custom start times and durations with millisecond precision
- **Adaptive Quality Settings**: Configurable resolution scaling (0.1x to 1.0x) and frame skip options
- **Real-time Progress Tracking**: Live progress updates during video processing operations
- **Large File Support**: Handles video files up to 10GB with built-in validation
- **Format Compatibility**: Support for MP4, MOV, AVI, and MKV video formats

### 🚀 Hardcore Mode Analysis (hardcore.py)
- **10-Agent Swarm System**: Parallel execution of specialized code analysis agents
- **API Documentation Diver**: Live API documentation lookup for all function calls
- **Architect Agent**: Structural analysis and MVC pattern recommendations
- **Security Auditor**: Automated security vulnerability detection
- **Performance Profiler**: Code optimization suggestions
- **Code Poet**: Code quality scoring and type hint recommendations
- **Test Engineer**: Automated test case generation
- **Future Proofer**: Python version compatibility and type safety checks
- **Feature Innovator**: New feature suggestions based on code analysis
- **Cost Optimizer**: API call optimization and cost reduction analysis
- **Synthesis Engine**: Merges all agent outputs into 9.9/10 quality code
- **Diff Generation**: Unified diff output showing before/after improvements
- **Proof of Work**: Detailed logs and metrics for all agent analyses

### 🎛️ Interactive Controls
- **Frame-by-Frame Navigation**: Arrow key controls for precise video positioning
- **Quick Timeline Setting**: Keyboard shortcuts (F/G keys) for rapid start time configuration
- **Dynamic Opacity Control**: Real-time alpha blending adjustment (0.1 to 1.0)
- **Responsive Design**: Optimized for both desktop and mobile interfaces
- **Theme Switching**: Light/dark mode toggle with persistent preferences

## Technical Architecture

### Project Structure
```
cipher/
├── main.py                 # Main Gradio application
├── hardcore.py             # 10-agent hardcore analysis system
├── static/                # Static assets
├── attached_assets/       # User-uploaded assets
└── replit.md             # This documentation file
```

### Frontend
- **Framework**: Gradio-based web interface with custom CSS styling
- **Responsive Design**: Mobile-first approach with adaptive layouts
- **Interactive Elements**: Custom JavaScript for video controls and keyboard shortcuts
- **Theme System**: Dual-theme support with CSS custom properties
- **Progress Visualization**: Real-time processing feedback with progress bars

### Backend
- **Core Framework**: Python with OpenCV for video processing
- **AI Integration**: Direct integration with xAI's Grok API endpoints
- **Video Processing**: Advanced computer vision algorithms for frame alignment
- **Memory Management**: Optimized for large file processing with batch operations
- **Error Handling**: Comprehensive validation and graceful error recovery
- **Logging System**: Detailed logging for debugging and monitoring
- **Multi-Agent Systems**: Parallel agent execution with ThreadPoolExecutor

### Performance Optimizations
- **Parallel Processing**: Multi-threaded video frame processing
- **Efficient Memory Usage**: Batch processing to minimize memory footprint
- **Smart Caching**: Previous transformation matrix caching for smoother alignment
- **Adaptive Algorithms**: ECC fallback mechanisms for challenging video content
- **Resource Management**: Automatic cleanup and resource deallocation
- **Agent Parallelization**: Concurrent execution of analysis agents

## API Integrations

### xAI Grok API
- **Text Model**: `grok-code-fast-1` for rapid code analysis
- **Vision Model**: `grok-4-0709` for image and visual content analysis
- **Streaming Support**: Server-sent events for real-time response delivery
- **Automatic Model Selection**: Dynamic switching based on input content type
- **Secure Authentication**: Environment-based API key management

## Configuration & Deployment

### Environment Setup
- **Python 3.11+**: Modern Python runtime with asyncio support
- **FFmpeg Integration**: Full multimedia processing capabilities via Nix packages
- **Port Configuration**: Configurable port binding (default: 5000)
- **SSL Support**: HTTPS-ready deployment configuration
- **Resource Limits**: Configurable thread limits and processing constraints

### Environment Variables
- `XAI_API_KEY`: Required for AI functionality (code analysis, hardcore mode)
- `PORT`: Server port (default: 5000)

### Security Features
- **Input Validation**: Comprehensive input sanitization and bounds checking
- **File Size Limits**: Configurable maximum file size enforcement (10GB default)
- **Secure File Handling**: Path validation and access control
- **API Key Protection**: Secure environment variable handling
- **Error Isolation**: Sandboxed error handling to prevent information leakage

### Monitoring & Debugging
- **Comprehensive Logging**: Multi-level logging with timestamp and context
- **Performance Metrics**: Frame processing statistics and timing information
- **Error Reporting**: Detailed error messages with context preservation
- **Progress Tracking**: Real-time processing status updates
- **Agent Execution Logs**: Detailed logs of all agent analyses and outputs

## User Interface Features

### Code Analysis Tab
- **Chat Interface**: Conversational AI interaction with syntax highlighting
- **Image URL Detection**: Automatic extraction and processing of image URLs
- **Response Streaming**: Real-time response generation with typing indicators
- **History Management**: Persistent conversation context across sessions

### Video Processing Tab
- **Dual Video Upload**: Side-by-side video preview and upload
- **Timeline Controls**: Precise timecode input (HH:MM:SS.mmm format)
- **Processing Options**: Advanced settings for quality, performance, and output control
- **Real-time Preview**: Live feedback during processing operations
- **Download Management**: Secure file download with progress tracking

### Accessibility Features
- **Keyboard Navigation**: Full keyboard support for all major functions
- **Responsive Layout**: Adaptive design for various screen sizes
- **Clear Visual Feedback**: Intuitive progress indicators and status messages
- **Error Communication**: User-friendly error messages with actionable guidance

## System Requirements

### Runtime Dependencies
- **Python**: 3.11 or higher
- **OpenCV**: 4.12.0+ with FFmpeg support
- **Gradio**: 5.44.1+ for web interface
- **OpenAI Client**: 1.102.0+ for API communication
- **Requests**: For HTTP API documentation lookups
- **NumPy**: For video frame processing

### System Dependencies
- **FFmpeg**: Full multimedia processing support
- **Nix Environment**: Stable package management and reproducible builds
- **Network Access**: HTTPS connectivity for API communications

## Advanced Features Usage

### Hardcore Mode
The hardcore.py module provides 10 specialized agents for comprehensive code analysis:
```python
from hardcore import run_hardcore_mode

# Analyze code at 9.9/10 intensity
final_code, diff, proof = run_hardcore_mode(original_code, intensity=9.9)
```

Returns:
- **final_code**: Improved code after all agent analyses
- **diff**: Unified diff showing all changes
- **proof**: Detailed metrics and logs from all 10 agents

## User Preferences

**Communication Style**: Simple, everyday language with clear explanations
**Interface Design**: Clean, minimalistic UI with focus on functionality
**Performance Priority**: Optimized for speed while maintaining quality output
**Code Quality**: Emphasis on security, performance, and maintainability
