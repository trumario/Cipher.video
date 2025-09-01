import os
import json
import re
import cv2
import subprocess
import requests
import gradio as gr
from openai import OpenAI
from typing import List, Dict, Any, Union, Optional, Tuple, Generator, cast
from functools import wraps
import time

# Constants
XAI_API_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-code-fast-1"
VISION_MODEL = "grok-4-0709"
API_KEY_PLACEHOLDER = "your_xai_api_key_here"
DEFAULT_PORT = 5000
MAX_PORT = 65535
MIN_PORT = 1
MAX_FILE_SIZE_GB = "5gb"  # Set to 5GB for better upload reliability
MAX_THREADS = 20
ALPHA_MIN = 0.1
ALPHA_MAX = 1.0
MAX_TOKENS = 2000
TEMPERATURE = 0.7

# Configuration - Load API key from environment variables
XAI_API_KEY = os.getenv("XAI_API_KEY", API_KEY_PLACEHOLDER)

def health_check() -> Dict[str, Union[str, float]]:
    """Simple health check endpoint for deployment"""
    return {"status": "healthy", "timestamp": time.time(), "service": "grok-chat-agent"}

def create_xai_client() -> Optional[OpenAI]:
    """Create xAI client with proper error handling"""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key or api_key == API_KEY_PLACEHOLDER:
        print("Warning: XAI_API_KEY not found or using placeholder value")
        print("App will continue to run but AI functionality will be limited")
        return None

    try:
        client = OpenAI(
            base_url=XAI_API_BASE_URL,
            api_key=api_key
        )
        try:
            # This is a minimal test that won't count against usage
            client.models.list()
            print("API key validation successful")
        except Exception as test_error:
            print(f"API key validation failed: {test_error}")
            print("App will continue to run but AI functionality may be limited")
        return client
    except Exception as e:
        print(f"Error creating xAI client: {e}")
        print("App will continue to run but AI functionality will be limited")
        return None

# Create xAI client
client = create_xai_client()

def extract_image_url(message: str) -> Optional[str]:
    """Extract image URLs from user message using regex"""
    # Improved regex for better security and accuracy
    url_pattern = re.compile(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s]*)?', re.IGNORECASE)
    match = url_pattern.search(message)
    return match.group(0) if match else None

def query_grok_streaming(user_input: str, history: Optional[List] = None, model: str = DEFAULT_MODEL, image_url: Optional[str] = None) -> Generator[str, None, None]:
    """Query Grok API with streaming response support"""
    if history is None:
        history = []

    if client is None:
        yield "Error: API client not available. Please set your XAI_API_KEY in the environment variables to enable AI functionality. The application is running but AI features are disabled."
        return

    try:
        messages = [
            {
                "role": "system", 
                "content": "You are an expert developer proficient in multiple programming languages, with extensive experience in secure coding practices, performance optimization, and code refactoring across various paradigms (e.g., procedural, object-oriented, functional). Your goal is to review and refactor the provided code to ensure it meets the highest standards of quality, security, and efficiency, tailored to the specified or inferred language.\n\nStep-by-Step Instructions:\n1. Understand the Codebase Context: Analyze the provided code in the context of the broader codebase. Identify opportunities to leverage existing base layer components, functions, or modules instead of reinventing functionality.\n2. Security Audit: Conduct a thorough security review. Check for vulnerabilities such as injection risks, improper input validation, authentication/authorization flaws, sensitive data exposure, and resource management issues.\n3. Remove Redundancy: Identify and eliminate redundant code, including duplicated logic, unused variables, or unnecessary computations.\n4. Eliminate TODOs and Placeholders: Remove all TODO comments, FIXMEs, or incomplete sections.\n5. Replace Magic Numbers and Hardcoded Values: Replace them with named constants or enums.\n6. Optimize Performance: Replace slow algorithms with optimized alternatives. Use efficient data structures and apply language-specific optimization techniques.\n\nProvide the fully refactored code in a single, complete block, followed by a concise summary of changes made. Do not introduce new features; only refine the existing code."
            }
        ]

        for human, ai in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": ai})

        if image_url:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        }
                    }
                ]
            }
            messages.append(user_message)
            model = VISION_MODEL
        else:
            messages.append({"role": "user", "content": user_input})

        try:
            # Cast messages to the proper type for OpenAI API
            formatted_messages = cast(Any, messages)
            stream = client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True
            )
        except Exception as api_error:
            yield f"API Error: {str(api_error)}"
            return

        partial_message = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                partial_message += chunk.choices[0].delta.content
                yield partial_message

    except Exception as e:
        yield f"Error communicating with Grok API: {str(e)}"
        return

def chat_function(message: str, history: List) -> str:
    """Main chat function for Gradio interface"""
    image_url = extract_image_url(message)

    # Collect all streaming responses into a single string
    full_response = ""
    for partial_response in query_grok_streaming(message, history, image_url=image_url):
        full_response = partial_response  # Each yield contains the complete message so far

    return full_response

def overlay_videos(base_path: str, ghost_path:

Optional[float] = None) -> Tuple[Optional[str], str]:
    """Overlay two videos with customizable parameters"""
    try:
        cap_base = cv2.VideoCapture(base_path)
        cap_ghost = cv2.VideoCapture(ghost_path)

        if not cap_base.isOpened() or not cap_ghost.isOpened():
            return None, "Error: Could not open one or both video files."

        cap_base.set(cv2.CAP_PROP_POS_MSEC, base_start_sec * 1000)
        cap_ghost.set(cv2.CAP_PROP_POS_MSEC, ghost_start_sec * 1000)

        fps = cap_base.get(cv2.CAP_PROP_FPS)
        width = int(cap_base.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_base.get(cv2.CAP_PROP_FRAME_HEIGHT))

        max_frames = int(duration_sec * fps) if duration_sec else None

        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            return None, "Error: Could not initialize video writer."

        processed_frames = 0

        while cap_base.isOpened() and cap_ghost.isOpened():
            if max_frames is not None and processed_frames >= max_frames:
                break

            ret_base, frame_base = cap_base.read()
            ret_ghost, frame_ghost = cap_ghost.read()

            if not ret_base or not ret_ghost:
                break

            if frame_ghost.shape != frame_base.shape:
                frame_ghost = cv2.resize(frame_ghost, (width, height))

            blended = cv2.addWeighted(frame_base, 1.0 - alpha, frame_ghost, alpha, 0)

            out.write(blended)
            processed_frames += 1

        cap_base.release()
        cap_ghost.release()
        out.release()

        return output_path, f"Successfully processed {processed_frames} frames. Video saved to: {output_path}"

    except Exception as e:
        return None, f"Video processing failed: {str(e)}"

def process_video_overlay(base_upload: str, ghost_upload: str, alpha: float, base_start: float, ghost_start: float, duration: Optional[float]) -> Tuple[Optional[str], Optional[str], str]:
    """Process video overlay with user inputs"""
    if not base_upload or not ghost_upload:
        return None, None, "Please upload both base and ghost videos."

    if not 0.1 <= alpha <= 1.0:
        return None, None, "Alpha value must be between 0.1 and 1.0"

    timestamp = int(time.time())
    output_path = f"overlay_output_{timestamp}.mp4"

    base_start = max(0.0, base_start)
    ghost_start = max(0.0, ghost_start)

    duration = duration if duration and duration > 0 else None

    result_path, status_msg = overlay_videos(
        base_upload, 
        ghost_upload, 
        output_path, 
        alpha, 
        base_start, 
        ghost_start, 
        duration
    )

    return result_path, output_path, status_msg

# Custom CSS for
# Custom CSS for stealthy dark/light themes (dark default)
CUSTOM_CSS = """
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">

:root {
    --primary-color: #4a4a4a;
    --bg-color: #1e1e1e;
    --text-color: #d4d4d4;
    --input-bg: #2a2a2a;
    --button-bg: #3a3a3a;
    --button-hover: #4a4a4a;
    --border-color: #3a3a3a;
    --accent-color: #666666;
}

.light {
    --primary-color: #007bff;
    --bg-color: #ffffff;
    --text-color: #333333;
    --input-bg: #f8f9fa;
    --button-bg: #e9ecef;
    --button-hover: #dee2e6;
    --border-color: #ced4da;
    --accent-color: #007bff;
}

/* Full screen layout */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    height: 100vh !important;
    overflow-x: hidden !important;
    background-color: var(--bg-color) !important;
    color: var(--text-color) !important;
    font-family: 'Arial', sans-serif;
    transition: all 0.3s ease;
    -webkit-text-size-adjust: 100%;
    -ms-text-size-adjust: 100%;
}

.gradio-container {
    background-color: var(--bg-color) !important;
    color: var(--text-color) !important;
    border: none !important;
    max-width: none !important;
    width: 100vw !important;
    height: 100vh !important;
    margin: 0 !important;
    padding: 10px !important;
    box-sizing: border-box !important;
}

/* Remove all white backgrounds and boxes */
.gr-form, .gr-box, .gr-panel, .gr-block, .gr-padded, .gr-rounded, .gradio-container > div {
    background-color: var(--bg-color) !important;
    border: none !important;
    box-shadow: none !important;
}

/* Input fields styling */
input, textarea, .gr-textbox input, .gr-number input, .gr-textbox textarea {
    background-color: var(--input-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    box-shadow: none !important;
    font-size: 16px !important;
    padding: 12px !important;
    -webkit-user-select: text !important;
    -moz-user-select: text !important;
    -ms-user-select: text !important;
    user-select: text !important;
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    appearance: none !important;
}

/* Button styling */
button, .gr-button {
    background-color: var(--button-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    transition: background-color 0.2s ease;
    box-shadow: none !important;
}

button:hover, .gr-button:hover {
    background-color: var(--button-hover) !important;
}

/* Tab styling */
.gr-tabitem, .tab-item {
    background-color: var(--bg-color) !important;
    color: var(--text-color) !important;
    border: none !important;
}

.tab-nav button, .tabs button {
    background-color: var(--button-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: none !important;
}

.tab-nav button.selected, .tabs button.selected {
    background-color: var(--accent-color) !important;
    color: #ffffff !important;
}

/* Chat interface styling */
.gr-chatbot, .chatbot {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    height: 70vh !important;
    box-shadow: none !important;
}

.gr-chatbot .message, .chatbot .message {
    background-color: var(--input-bg) !important;
    color: var(--text-color) !important;
    border: none !important;
}

/* Video components styling */
.gr-video, .video-container {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}

/* Slider styling */
.gr-slider, .gr-slider input[type="range"] {
    background-color: var(--input-bg) !important;
}

.gr-slider .gr-slider-track {
    background-color: var(--border-color) !important;
}

/* Markdown styling */
.gr-markdown, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: var(--text-color) !important;
    background-color: transparent !important;
}

/* Remove any white boxes and backgrounds */
.white, .bg-white, .bg-gray-50, .bg-gray-100 {
    background-color: var(--bg-color) !important;
}

/* Ensure all containers are dark themed */
div, span, p, label {
    color: var(--text-color) !important;
}

/* Mobile responsive design */
@media (max-width: 768px) {
    html, body {
        -webkit-text-size-adjust: none !important;
        touch-action: manipulation !important;
    }
    
    .gradio-container {
        padding: 8px !important;
        width: 100vw !important;
        height: 100vh !important;
        overflow-x: hidden !important;
    }
    
    .gr-chatbot, .chatbot {
        height: 55vh !important;
        min-height: 300px !important;
    }
    
    .gr-row {
        flex-direction: column !important;
        gap: 8px !important;
    }
    
    .gr-column {
        width: 100% !important;
        margin-bottom: 8px !important;
        min-width: 0 !important;
    }
    
    /* Enhanced input handling for mobile */
    input, textarea, .gr-textbox input, .gr-textbox textarea {
        font-size: 16px !important;
        padding: 14px 12px !important;
        min-height: 44px !important;
        border-radius: 8px !important;
        touch-action: manipulation !important;
        -webkit-touch-callout: default !important;
        -webkit-user-select: text !important;
    }
    
    button, .gr-button {
        min-height: 44px !important;
        padding: 12px 16px !important;
        font-size: 16px !important;
        touch-action: manipulation !important;
    }
    
    .gr-video, .video-container {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
    }
    
    /* Prevent zoom on input focus */
    @media screen and (-webkit-min-device-pixel-ratio: 0) {
        select, textarea, input[type="text"], input[type="password"],
        input[type="datetime"], input[type="datetime-local"],
        input[type="date"], input[type="month"], input[type="time"],
        input[type="week"], input[type="number"], input[type="email"],
        input[type="url"], input[type="search"], input[type="tel"],
        input[type="color"] {
            font-size: 16px !important;
        }
    }
}

/* Desktop full screen */
@media (min-width: 769px) {
    .gradio-container {
        width: 100vw !important;
        height: 100vh !important;
        max-width: none !important;
    }
    
    .gr-chatbot, .chatbot {
        height: 75vh !important;
    }
}

/* Remove any emoji or decorative elements */
.emoji, .fun-icon { 
    display: none !important; 
}

/* Theme toggle button styling */
.theme-toggle {
    width: 40px !important;
    height: 30px !important;
    padding: 4px !important;
    font-size: 16px !important;
    min-width: 40px !important;
    border-radius: 6px !important;
    background-color: var(--button-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
}

/* Ensure all text is visible */
* {
    color: var(--text-color) !important;
}

/* Force dark theme on all elements */
*, *::before, *::after {
    background-color: var(--bg-color) !important;
    border-color: var(--border-color) !important;
}

/* Exceptions for input elements */
input, textarea, button, .gr-video, .gr-chatbot {
    background-color: var(--input-bg) !important;
}
"""

# Health check function available for deployment monitoring

# Create Gradio interface
with gr.Blocks(
    title="Cipher",
    css=CUSTOM_CSS
) as demo:
    
    with gr.Row():
        with gr.Column(scale=9):
            gr.Markdown(
                """<h1 style="font-family: 'Courier New', monospace; font-weight: bold; color: var(--text-color); text-transform: uppercase; letter-spacing: 3px; margin: 0; text-shadow: 0 0 10px var(--accent-color);">CIPHER</h1>"""
            )
        with gr.Column(scale=1, min_width=80):
            toggle_btn = gr.Button("◐", size="sm", elem_classes=["theme-toggle"])
            toggle_btn.click(None, js="""() => {
                document.body.classList.toggle('light');
                return null;
            }""")
    
    with gr.Tab("Code"):
        chat_interface = gr.ChatInterface(
            chat_function,
            type="messages",
            textbox=gr.Textbox(
                placeholder="Enter code or image URL...",
                container=False
            )
        )
    
    with gr.Tab("Video"):
        with gr.Row():
            base_upload = gr.Video(label="Base")
            ghost_upload = gr.Video(label="Ghost")
        with gr.Row():
            alpha_slider = gr.Slider(0.1, 1.0, value=0.5, label="Opacity")
            base_start = gr.Number(value=0.0, label="Base Start")
            ghost_start = gr.Number(value=0.0, label="Ghost Start")
            duration = gr.Number(value=None, label="Duration")
        process_btn = gr.Button("Process")
        output_video = gr.Video(label="Output")
        save_location = gr.Textbox(label="Save Location", interactive=False)
        status_output = gr.Textbox(label="Status", interactive=False)
        
        process_btn.click(
            fn=process_video_overlay,
            inputs=[base_upload, ghost_upload, alpha_slider, base_start, ghost_start, duration],
            outputs=[output_video, save_location, status_output]
        )

# Note: Gradio will automatically handle the root endpoint for health checks
# The demo.launch() will make the app respond to requests at the root path

# Launch the application
if __name__ == "__main__":
    try:
        # Get port from environment variable for deployment compatibility
        try:
            port = int(os.getenv("PORT", "5000"))
        except (ValueError, TypeError):
            print("Warning: Invalid PORT environment variable, using default port 5000")
            port = 5000
        
        # Validate port range
        if not (1 <= port <= 65535):
            print(f"Warning: Invalid port {port}, using default port 5000")
            port = 5000
        
        # Check optional but important environment variables
        xai_key = os.getenv("XAI_API_KEY")
        if not xai_key or xai_key == "your_xai_api_key_here":
            print("Warning: XAI_API_KEY not properly configured. AI functionality will be limited.")
            print("To enable full functionality, set your XAI_API_KEY in the environment variables.")
        
        # Validate that all required dependencies are available
        try:
            import cv2
            import gradio
            import openai
            print("All required dependencies are available")
        except ImportError as ie:
            print(f"Missing required dependency: {ie}")
            print("Please ensure all dependencies are properly installed")
        
        print(f"Starting Gradio app on port {port}")
        print("API client status:", "Connected" if client else "Not connected (app will still run)")
        print(f"Application health check available at: http://0.0.0.0:{port}/")
        
        # Health check will be handled by Gradio's default endpoints
        print("Health check available at root endpoint")
        
        # Configure Gradio for deployment with enhanced settings
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
            quiet=False,
            favicon_path=None,
            ssl_verify=False,
            max_threads=20,  # Increased for better deployment performance
            inbrowser=False,  # Disable auto-opening browser in deployment
            root_path=None,  # Let deployment infrastructure handle root path
            max_file_size="5gb",  # Increased to 5GB for video uploads
            app_kwargs={
                "docs_url": None,  # Disable Swagger docs in production
                "redoc_url": None  # Disable ReDoc in production
            }
        )
        
    except ValueError as ve:
        print(f"Configuration error: {ve}")
        print("Please check your environment variables and configuration.")
        exit(1)
    except Exception as e:
        print(f"Failed to start application: {e}")
        print("Please check your configuration and try again.")
        exit(1)
