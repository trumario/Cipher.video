
# Cipher Code - AI-Powered Code Analysis & Video Processing Platform

A sophisticated web application that combines AI-powered code analysis with advanced video processing capabilities. Built with Python and Gradio, featuring integration with xAI's Grok API for intelligent code review across three specialized modes and a powerful video overlay system for precision video editing.

## Core Features

### 🤖 AI Code Analysis - Three Specialized Modes

#### Mode 1: LEARNING (Patient Mentor)
A senior software engineer and patient mentor that helps you become a better programmer:
- **Clarification First**: Confirms language, environment, constraints, and edge cases before proceeding
- **Step-by-Step Teaching**: Breaks problems into small, manageable steps
- **Plain English Explanations**: Every line explained in simple, everyday language
- **Official Documentation**: Cites official docs with URLs for every API/method used
- **Test Cases**: Shows input-to-expected output examples for verification
- **Concept Highlighting**: Emphasizes one key learning concept per step
- **Interactive Feedback**: Asks "Does this make sense? Ready for the next step?"
- **No Assumptions**: Always asks for clarification when context is missing
- **Runnable Tests**: Includes assertions and local verification instructions

**Teaching Process:**
1. Clarify requirements (language, environment, constraints, edge cases)
2. Show small, complete code blocks
3. Explain every line in plain English
4. Cite official documentation with URLs
5. Provide test cases
6. Highlight key concepts
7. Ask for confirmation before proceeding

#### Mode 2: POLISH (Production-Ready Refinement)
Principal engineer approach for Fortune 500-grade code:
- **Security First**: Eliminates path traversal, injection, overflow, race conditions
- **Performance Optimization**: O(n log n) or better algorithms, minimal allocations
- **Clean Code**: No magic numbers, TODOs, or redundancy; named constants and single responsibility
- **Type Safety**: Full type hints and comprehensive validation
- **Reliability**: Input validation, graceful degradation, resource cleanup, logging
- **Standards Compliance**: Adheres to official coding standards (PEP 8 & PEP 20 for Python)
- **Line-by-Line Justification**: Explains changes by line number
- **Production-Ready Output**: Delivers clean, enterprise-grade code

**Refinement Rules:**
1. Clarify requirements first
2. Eliminate all security vulnerabilities
3. Optimize for performance
4. Remove all code smells
5. Add comprehensive validation
6. Follow language-specific standards
7. Explain all changes with line numbers

#### Mode 3: HARDCORE (10-Agent Swarm System)
Comprehensive code analysis through parallel execution of specialized agents:
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

**HARDCORE Process:**
1. Parse code and extract all function calls
2. Run 10 agents in parallel (ThreadPoolExecutor)
3. API Doc Diver fetches live documentation
4. Security Auditor scans for vulnerabilities
5. Performance Profiler suggests optimizations
6. All agents provide specialized analysis
7. Synthesis engine merges outputs
8. Generate unified diff showing improvements
9. Return 9.9/10 code with proof of work

### 🎥 Advanced Video Processing
- **Precision Video Overlay**: ECC (Enhanced Correlation Coefficient) based frame alignment and stabilization
- **Multi-threaded Processing**: Optimized batch frame processing using ThreadPoolExecutor
- **Millisecond Precision**: HH:MM:SS.mmm format for timeline control (e.g., 00:01:23.500)
- **Adaptive Quality Settings**: 
  - Resolution scaling: 0.1x to 1.0x (lower = faster processing)
  - Frame skip: 1 to 10 (higher = skip more frames, faster but lower quality)
- **Real-time Progress Tracking**: Live progress updates during video processing operations
- **Large File Support**: Handles video files up to 10GB with built-in validation
- **Format Compatibility**: Support for MP4, MOV, AVI, and MKV video formats
- **Frame Navigation**: Arrow key controls for frame-by-frame video positioning
- **Quick Timeline Setting**: Keyboard shortcuts (F/G keys) for rapid start time configuration
- **Alpha Blending**: Dynamic opacity control (0.1 to 1.0) for ghost video transparency
- **Download Management**: Secure file download with progress tracking

### 📎 File Upload & Multi-modal Input
- **Image Support**: JPG, JPEG, PNG, GIF, WebP (up to 50MB)
- **Code File Support**: Python, JavaScript, C/C++, Java, HTML, CSS, JSON, XML, Markdown, YAML, Shell scripts (up to 10MB)
- **Vision Analysis**: Automatic image analysis using Grok vision model
- **Attach Button**: Quick file upload via 📎 button
- **Multi-modal Chat**: Seamlessly mix text, code, and images in conversations

### 🎛️ Interactive Controls
- **Mode Toggle**: Click to cycle through LEARNING → POLISH → HARDCORE
- **Frame-by-Frame Navigation**: Arrow key controls for precise video positioning
- **Quick Timeline Setting**: F key for Base video, G key for Ghost video
- **Dynamic Opacity Control**: Real-time alpha blending adjustment (0.1 to 1.0)
- **Responsive Design**: Optimized for both desktop and mobile interfaces
- **Theme Switching**: Light/dark mode toggle with half-circle button

## Technical Architecture

### Project Structure
```
cipher/
├── main.py                 # Main Gradio application (823 lines)
├── hardcore.py             # 10-agent hardcore analysis system
├── static/                # Static assets
├── attached_assets/       # User-uploaded assets
├── README.md             # User-facing documentation
└── replit.md             # Technical documentation (this file)
```

### Frontend
- **Framework**: Gradio-based web interface with custom CSS styling
- **Responsive Design**: Mobile-first approach with adaptive layouts
- **Interactive Elements**: Custom JavaScript for video controls and keyboard shortcuts
- **Theme System**: Dual-theme support (light/dark) with CSS custom properties
- **Progress Visualization**: Real-time processing feedback with progress bars
- **File Upload**: UploadButton component with file type restrictions

### Backend
- **Core Framework**: Python 3.11+ with asyncio support
- **AI Integration**: Direct integration with xAI's Grok API endpoints
- **Video Processing**: OpenCV 4.12.0+ with advanced computer vision algorithms
- **Memory Management**: Optimized for large file processing with batch operations
- **Error Handling**: Comprehensive validation and graceful error recovery
- **Logging System**: Multi-level logging with timestamp and context
- **Multi-Agent Systems**: Parallel agent execution with ThreadPoolExecutor
- **Secure File Handling**: Temporary directory with automatic cleanup on exit

### Performance Optimizations
- **Parallel Processing**: Multi-threaded video frame processing (up to 8 threads)
- **Efficient Memory Usage**: Batch processing to minimize memory footprint
- **Smart Caching**: Previous transformation matrix caching for smoother alignment
- **Adaptive Algorithms**: ECC fallback mechanisms for challenging video content
- **Resource Management**: Automatic cleanup and resource deallocation via atexit
- **Agent Parallelization**: Concurrent execution of all 10 HARDCORE agents
- **Progress Updates**: Updates every 100 frames to minimize overhead

## API Integrations

### xAI Grok API
- **Text Model**: `grok-code-fast-1` for rapid code analysis (LEARNING & POLISH modes)
- **Vision Model**: `grok-4-fast-reasoning` for image analysis and advanced reasoning
- **Streaming Support**: Server-sent events for real-time response delivery
- **Automatic Model Selection**: Dynamic switching based on input content type
- **Secure Authentication**: Environment-based API key management
- **Error Handling**: Comprehensive error catching for AuthenticationError and APIError

## Configuration & Deployment

### Environment Setup
- **Python 3.11+**: Modern Python runtime with asyncio support
- **FFmpeg Integration**: Full multimedia processing capabilities via Nix packages
- **Port Configuration**: Configurable port binding (default: 5000)
- **SSL Support**: HTTPS-ready deployment configuration with ssl_verify=False
- **Resource Limits**: Configurable thread limits (default: 8, max: 20)
- **Temporary Storage**: Auto-generated temp directory with cleanup

### Environment Variables
- `XAI_API_KEY`: Required for AI functionality (code analysis in all modes)
- `PORT`: Server port (default: 5000, range: 1-65535)

### Constants & Limits
| Constant | Value | Purpose |
|----------|-------|---------|
| MAX_FILE_SIZE_GB | 10 | Maximum video file size |
| MAX_IMAGE_FILE_SIZE_BYTES | 50MB | Maximum image file size |
| MAX_CODE_FILE_SIZE_BYTES | 10MB | Maximum code file size |
| MAX_THREADS | 8 | Maximum concurrent threads |
| DEFAULT_FRAME_SKIP | 5 | Default frames to skip |
| MAX_FRAME_SKIP | 10 | Maximum frames to skip |
| DEFAULT_RESOLUTION_SCALE | 0.5 | Default resolution scale |
| ALPHA_MIN / ALPHA_MAX | 0.1 / 1.0 | Opacity range |
| ECC_ITERATIONS | 50 | Motion estimation iterations |
| ECC_EPSILON | 1e-10 | Convergence threshold |
| MAX_COMPLETION_TOKENS | 16384 | Max AI response tokens |

### Security Features
- **Input Validation**: Comprehensive input sanitization and bounds checking
- **File Size Limits**: Enforced maximum file sizes per type
- **Secure File Handling**: Path validation with `pathlib.resolve()` and `is_relative_to()`
- **API Key Protection**: Secure environment variable handling
- **Error Isolation**: Sandboxed error handling to prevent information leakage
- **Path Traversal Protection**: All file operations restricted to ALLOWED_BASE_DIR
- **Temporary Storage**: Isolated upload directory with automatic cleanup

### Monitoring & Debugging
- **Comprehensive Logging**: Multi-level logging (INFO, WARNING, ERROR)
- **Performance Metrics**: Frame processing statistics and timing information
- **Error Reporting**: Detailed error messages with context preservation
- **Progress Tracking**: Real-time processing status updates
- **Agent Execution Logs**: Detailed logs of all agent analyses and outputs
- **API Status Logging**: Connection status and validation results

## User Interface Features

### Code Tab
- **Mode Selector**: Button to toggle between LEARNING, POLISH, and HARDCORE modes
- **Chat Interface**: Conversational AI interaction with message history
- **File Attachment**: 📎 button for uploading images and code files
- **Response Streaming**: Real-time response generation with typing indicators
- **History Management**: Persistent conversation context across sessions
- **Multi-line Input**: 5-line text area with auto-resize
- **Submit Button**: "Code" button to send messages

### Video Processing Tab
- **Dual Video Upload**: Side-by-side video preview and upload (Base + Ghost)
- **Timeline Controls**: Precise HH:MM:SS.mmm format input fields
- **Set from Position Buttons**: Quick-set current video position to start time
- **Processing Options**: 
  - Opacity slider (0.1 to 1.0)
  - Duration input (optional, HH:MM:SS.mmm)
  - Frame skip selector (1 to 10)
  - Resolution scale slider (0.1 to 1.0)
- **Process Button**: Initiates video overlay processing
- **Output Video Player**: Preview processed video
- **Download Button**: Download processed video file
- **Status Output**: Real-time processing status and error messages
- **Save Location**: Display of output file path

### Keyboard Shortcuts
- **Arrow Left/Right**: Navigate video frame-by-frame (after clicking video to focus)
- **F Key**: Set Base Start time from current Base video position
- **G Key**: Set Ghost Start time from current Ghost video position

### Accessibility Features
- **Keyboard Navigation**: Full keyboard support for all major functions
- **Responsive Layout**: Adaptive design for various screen sizes
- **Clear Visual Feedback**: Intuitive progress indicators and status messages
- **Error Communication**: User-friendly error messages with actionable guidance
- **Focus Management**: Proper focus handling for video elements
- **Mobile Optimization**: Touch-friendly controls and layouts

## System Requirements

### Runtime Dependencies
- **Python**: 3.11 or higher
- **OpenCV**: 4.12.0+ with FFmpeg support (opencv-python and opencv-python-headless)
- **Gradio**: 5.44.1+ for web interface
- **OpenAI Client**: 1.102.0+ for API communication
- **Requests**: For HTTP API documentation lookups (hardcore.py)
- **NumPy**: For video frame processing and array operations
- **FFmpeg-Python**: For multimedia processing

### System Dependencies
- **FFmpeg**: Full multimedia processing support
- **Nix Environment**: Stable package management and reproducible builds
- **Network Access**: HTTPS connectivity for API communications

## Advanced Features Usage

### LEARNING Mode
```python
# System prompt ensures:
# 1. Clarify First (language, environment, constraints, edge cases)
# 2. Teach As You Go (small steps, explanations, docs, tests, concepts)
# 3. Prove It Works (runnable tests, verification instructions)
# 4. Ask Before Moving On (confirmation questions)
# 5. No Assumptions (always ask if context is missing)
```

**Example workflow:**
1. User: "How do I parse a CSV file?"
2. AI asks: "What language? What's your environment? Any constraints?"
3. User confirms
4. AI shows Step 1 with code, explanations, docs, test
5. AI asks: "Does this make sense? Ready for the next step?"
6. Process continues until complete

### POLISH Mode
```python
# System prompt ensures:
# 1. Clarify First (confirm requirements)
# 2. Security (eliminate vulnerabilities)
# 3. Performance (O(n log n) or better)
# 4. Clean Code (no magic numbers, TODOs)
# 5. Reliability (validation, logging)
# 6. Standards (PEP 8, PEP 20 for Python)
```

**Example output:**
```
Line 15: Changed os.getcwd() to UPLOAD_DIR
Reason: Security - prevents path traversal attacks

Line 23: Added type hint: def process(data: List[str]) -> Dict[str, int]
Reason: Type safety and IDE support

Line 45: Replaced 100 with MAX_RETRIES constant
Reason: Clean code - eliminate magic numbers
```

### HARDCORE Mode
The hardcore.py module provides 10 specialized agents for comprehensive code analysis:
```python
from hardcore import run_hardcore_mode

# Analyze code at 9.9/10 intensity
final_code, diff, proof = run_hardcore_mode(original_code, intensity=9.9)

print(f"Score: {proof['score']}")
print(f"Agents ran: {proof['agents_ran']}")
print(f"Duration: {proof['duration']}")
print(f"Proof logs: {proof['proof_log']}")
print("\nDiff:\n", diff)
```

**Returns:**
- **final_code**: Improved code after all agent analyses
- **diff**: Unified diff showing all changes (from difflib)
- **proof**: Detailed metrics including:
  - `score`: Quality score (9.9/10)
  - `agents_ran`: Number of agents executed
  - `duration`: Total execution time
  - `proof_log`: Individual agent proofs

**Agent Breakdown:**
1. **API Doc Diver**: Parses AST, extracts function calls, fetches live docs from python.org, opencv.org, gradio.app
2. **Architect**: Suggests module structure (video/, ai/, ui/), MVC pattern
3. **Security Auditor**: Scans for os.getcwd(), unsafe open() calls
4. **Performance Profiler**: Identifies optimization opportunities (e.g., precompute frame indices)
5. **Code Poet**: Rates code quality, suggests type hints
6. **Test Engineer**: Generates test functions with assertions
7. **Future Proofer**: Checks Python 3.13 compatibility, suggests mypy
8. **Feature Innovator**: Proposes new features (e.g., "Add 'Export as GIF' button")
9. **Cost Optimizer**: Analyzes API call patterns, suggests optimizations
10. **Synthesis**: Merges all outputs, applies fixes, generates final code

## Video Processing Pipeline

### Processing Steps
1. **Input Validation**: Check file paths, sizes, and formats
2. **Video Capture**: Open both videos with cv2.VideoCapture
3. **Timeline Positioning**: Seek to specified start times (in milliseconds)
4. **FPS Detection**: Extract frames per second from base video
5. **Resolution Calculation**: Apply resolution scale to dimensions
6. **Frame Extraction**: Read frames with skip optimization
7. **ECC Alignment**: Apply Enhanced Correlation Coefficient motion estimation
8. **Alpha Blending**: Weighted overlay with cv2.addWeighted
9. **Batch Processing**: Multi-threaded processing with ThreadPoolExecutor
10. **H264 Encoding**: Write output with H264 codec
11. **Cleanup**: Release all video captures and writers

### ECC (Enhanced Correlation Coefficient)
- **Motion Model**: MOTION_HOMOGRAPHY for perspective transformations
- **Iterations**: 50 iterations for convergence
- **Epsilon**: 1e-10 convergence threshold
- **Fallback**: Identity matrix if ECC fails
- **Threshold**: 0.6 minimum correlation for successful alignment

### Performance Tips
- **Fast Preview**: Frame skip = 5-10, Resolution scale = 0.3-0.5
- **High Quality**: Frame skip = 1-2, Resolution scale = 0.8-1.0
- **Balanced**: Frame skip = 3-5, Resolution scale = 0.5-0.7

## User Preferences

**Communication Style**: Simple, everyday language with clear explanations
**Interface Design**: Clean, minimalistic UI with focus on functionality
**Performance Priority**: Optimized for speed while maintaining quality output
**Code Quality**: Emphasis on security, performance, and maintainability
**Learning Focus**: Patient teaching approach with step-by-step explanations
