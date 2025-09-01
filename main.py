import os
import json
import re
import cv2
import subprocess
import requests
import gradio as gr
from openai import OpenAI
from typing import List, Dict, Any, Union, Optional, Tuple, Generator
from functools import wraps
import time
import logging

# Set up logging for better error handling and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
XAI_API_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-code-fast-1"
VISION_MODEL = "grok-4-0709"
DEFAULT_PORT = 5000
MAX_PORT = 65535
MIN_PORT = 1
MAX_FILE_SIZE_GB = 5
MAX_THREADS = 20
ALPHA_MIN = 0.1
ALPHA_MAX = 1.0
MAX_TOKENS = 2000
TEMPERATURE = 0.7

# Configuration - Load API key from environment variables
XAI_API_KEY = os.getenv("XAI_API_KEY")

def health_check() -> Dict[str, Union[str, float]]:
    """Simple health check endpoint for deployment"""
    return {"status": "healthy", "timestamp": time.time(), "service": "grok-chat-agent"}

def create_xai_client() -> Optional[OpenAI]:
    """Create xAI client with proper error handling"""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        logger.warning("XAI_API_KEY not found. App will continue but AI functionality is limited.")
        return None
    try:
        client = OpenAI(base_url=XAI_API_BASE_URL, api_key=api_key)
        client.models.list()
        logger.info("API key validation successful")
    except Exception as e:
        logger.error(f"Error creating or validating xAI client: {e}")
        return None
    return client

# Create xAI client
client = create_xai_client()

def extract_image_url(message: str) -> Optional[str]:
    """Extract image URLs from user message using regex with input validation"""
    if not message or not isinstance(message, str):
        return None
    url_pattern = re.compile(
        r'https?://[^\s<>"\'{}]{1,2048}\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s<>"\'{}]{0,1024})?', 
        re.IGNORECASE
    )
    match = url_pattern.search(message)
    return match.group(0) if match else None

def query_grok_streaming(
    user_input: str, 
    history: Optional[List] = None, 
    model: str = DEFAULT_MODEL, 
    image_url: Optional[str] = None
) -> Generator[str, None, None]:
    """Query Grok API with streaming response support"""
    if history is None:
        history = []
    if client is None:
        yield "Error: API client not available. Please set your XAI_API_KEY."
        return
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert developer proficient in multiple programming languages, "
                    "with extensive experience in secure coding practices, performance optimization, "
                    "and code refactoring. Your goal is to review and refactor the provided code to "
                    "meet the highest standards of quality, security, and efficiency, tailored to the "
                    "specified or inferred language.\n\n"
                    "Steps:\n"
                    "1. Analyze the codebase context and leverage existing components.\n"
                    "2. Conduct a security audit for vulnerabilities.\n"
                    "3. Remove redundant code, unused variables, or unnecessary computations.\n"
                    "4. Eliminate TODOs and placeholders.\n"
                    "5. Replace magic numbers with named constants.\n"
                    "6. Optimize performance with efficient algorithms and data structures.\n"
                    "Provide the refactored code in a single block, followed by a summary of changes."
                )
            }
        ]
        for human, ai in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": ai})
        if image_url:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}}
                ]
            })
            model = VISION_MODEL
        else:
            messages.append({"role": "user", "content": user_input})

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True
        )
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
    full_response = ""
    for partial_response in query_grok_streaming(message, history, image_url=image_url):
        full_response = partial_response
    return full_response

def overlay_videos(
    base_path: str, 
    ghost_path: str, 
    output_path: str, 
    alpha: float, 
    base_start_sec: float, 
    ghost_start_sec: float, 
    duration_sec: Optional[float]
) -> Tuple[Optional[str], str]:
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

def process_video_overlay(
    base_upload: str, 
    ghost_upload: str, 
    alpha: float, 
    base_start: float, 
    ghost_start: float, 
    duration: Optional[float]
) -> Tuple[Optional[str], Optional[str], str]:
    """Process video overlay with user inputs"""
    if not base_upload or not ghost_upload:
        return None, None, "Please upload both base and ghost videos."
    if not ALPHA_MIN <= alpha <= ALPHA_MAX:
        return None, None, f"Alpha value must be between {ALPHA_MIN} and {ALPHA_MAX}"

    timestamp = int(time.time())
    output_path = f"overlay_output_{timestamp}.mp4"
    base_start = max(0.0, base_start)
    ghost_start = max(0.0, ghost_start)
    duration = duration if duration and duration > 0 else None

    result_path, status_msg = overlay_videos(
        base_path=base_upload,
        ghost_path=ghost_upload,
        output_path=output_path,
        alpha=alpha,
        base_start_sec=base_start,
        ghost_start_sec=ghost_start,
        duration_sec=duration
    )
    return result_path, output_path, status_msg

# Custom CSS for enhanced mobile UI and dark theme default
CUSTOM_CSS = """
/* Ensure proper viewport scaling */
meta[name="viewport"] {
    content: "width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no";
}

/* Theme variables */
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

/* Base styles */
html, body {
    margin: 0;
    padding: 0;
    height: 100vh;
    width: 100vw;
    overflow-x: hidden;
    background-color: var(--bg-color);
    color: var(--text-color);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* Gradio container */
.gradio-container {
    background-color: var(--bg-color);
    color: var(--text-color);
    max-width: 100vw;
    min-height: 100vh;
    margin: 0;
    padding: 8px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
}

/* Remove unwanted backgrounds */
.gr-form, .gr-box, .gr-panel, .gr-block, .gr-padded, .gr-rounded, .gradio-container > div {
    background-color: transparent;
    border: none;
    box-shadow: none;
}

/* Input and textarea styling */
input, textarea, .gr-textbox input, .gr-textbox textarea {
    background-color: var(--input-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    font-size: 16px;
    padding: 12px;
    box-sizing: border-box;
    width: 100%;
}

/* Button styling */
button, .gr-button {
    background-color: var(--button-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 16px;
    cursor: pointer;
    transition: background-color 0.2s ease;
}
button:hover, .gr-button:hover {
    background-color: var(--button-hover);
}

/* Chat interface */
.gr-chatbot, .chatbot {
    background-color: var(--input-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    flex: 1;
    min-height: 300px;
    overflow-y: auto;
}
.gr-chatbot .message, .chatbot .message {
    background-color: transparent;
    color: var(--text-color);
}

/* Video components */
.gr-video, .video-container {
    background-color: var(--input-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    width: 100%;
    height: auto;
}

/* Slider styling */
.gr-slider input[type="range"] {
    background-color: transparent;
    accent-color: var(--accent-color);
}

/* Markdown styling */
.gr-markdown, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: var(--text-color);
    background-color: transparent;
}

/* Theme toggle button */
.theme-toggle {
    width: 40px;
    height: 40px;
    padding: 8px;
    font-size: 18px;
    border-radius: 8px;
    background-color: var(--button-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
}

/* Mobile-specific styles */
@media (max-width: 768px) {
    html, body {
        -webkit-text-size-adjust: none;
        touch-action: manipulation;
    }
    .gradio-container {
        padding: 8px;
    }
    .gr-row {
        flex-direction: column;
        gap: 12px;
    }
    .gr-column {
        width: 100%;
        min-width: 0;
        margin-bottom: 12px;
    }
    input, textarea, .gr-textbox input, .gr-textbox textarea {
        font-size: 16px;
        padding: 14px;
        min-height: 48px;
        border-radius: 8px;
    }
    button, .gr-button {
        min-height: 48px;
        padding: 12px 16px;
        font-size: 16px;
    }
    .gr-chatbot, .chatbot {
        min-height: 50vh;
    }
    .gr-video, .video-container {
        max-width: 100%;
        height: auto;
    }
}

/* Desktop styles */
@media (min-width: 769px) {
    .gradio-container {
        padding: 16px;
    }
    .gr-chatbot, .chatbot {
        min-height: 70vh;
    }
}

/* Remove decorative elements */
.emoji, .fun-icon {
    display: none;
}
"""

# Create Gradio interface
with gr.Blocks(title="Cipher", css=CUSTOM_CSS) as demo:
    with gr.Row():
        with gr.Column(scale=9):
            gr.Markdown(
                """<h1 style="font-family: 'Courier New', monospace; font-weight: bold; color: var(--text-color); text-transform: uppercase; letter-spacing: 2px;">CIPHER</h1>"""
            )
        with gr.Column(scale=1, min_width=80):
            toggle_btn = gr.Button("◐", size="sm", elem_classes=["theme-toggle"])
            toggle_btn.click(None, js="""() => {
                document.body.classList.toggle('light');
                return null;
            }""")

    with gr.Tab("Code"):
        chat_interface = gr.ChatInterface(
            fn=chat_function,
            type="messages",
            textbox=gr.Textbox(placeholder="Enter code or image URL...", container=False)
        )

    with gr.Tab("Video"):
        with gr.Row():
            base_upload = gr.Video(label="Base Video")
            ghost_upload = gr.Video(label="Ghost Video")
        with gr.Row():
            alpha_slider = gr.Slider(ALPHA_MIN, ALPHA_MAX, value=0.5, label="Opacity")
            base_start = gr.Number(value=0.0, label="Base Start (s)")
            ghost_start = gr.Number(value=0.0, label="Ghost Start (s)")
            duration = gr.Number(value=None, label="Duration (s)")
        process_btn = gr.Button("Process")
        output_video = gr.Video(label="Output Video")
        save_location = gr.Textbox(label="Save Location", interactive=False)
        status_output = gr.Textbox(label="Status", interactive=False)

        process_btn.click(
            fn=process_video_overlay,
            inputs=[base_upload, ghost_upload, alpha_slider, base_start, ghost_start, duration],
            outputs=[output_video, save_location, status_output]
        )

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", DEFAULT_PORT))
        if not MIN_PORT <= port <= MAX_PORT:
            logger.warning(f"Invalid port {port}, using default port {DEFAULT_PORT}")
            port = DEFAULT_PORT

        if not XAI_API_KEY or XAI_API_KEY == "your_xai_api_key_here":
            logger.warning("XAI_API_KEY not properly configured. AI functionality will be limited.")

        logger.info(f"Starting Gradio app on port {port}")
        logger.info("API client status: %s", "Connected" if client else "Not connected")
        logger.info(f"Application health check available at: http://0.0.0.0:{port}/")

        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
            max_threads=MAX_THREADS,
            max_file_size=f"{MAX_FILE_SIZE_GB}gb",
            ssl_verify=False,
            inbrowser=False
        )
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        exit(1)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        exit(1)