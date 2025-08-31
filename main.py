import os
import re
import json
import cv2
import subprocess
import gradio as gr
from openai import OpenAI
from typing import List, Dict, Union, Any, Generator, Iterator
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Constants
XAI_API_KEY = os.getenv("XAI_API_KEY")
DEFAULT_MODEL = "grok-4-0709"
VISION_MODEL = "grok-4-0709"
MAX_TOKENS = 2000
TEMPERATURE = 0.7
OUTPUT_VIDEO_PATH = "output_overlay.mp4"
TEMP_VIDEO_SUFFIX = "_temp_video_only.mp4"
MIN_ALPHA = 0.1
MAX_ALPHA = 1.0
DEFAULT_ALPHA = 0.5
MIN_START_SEC = 0.0
VIDEO_DETAIL_LEVEL = "high"

if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not set in environment variables.")

# Create xAI client using OpenAI pattern
client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=XAI_API_KEY
)

def extract_image_url(message: str) -> Union[str, None]:
    """Extract the first image URL from user message using regex"""
    urls = re.findall(r'(https?://\S+\.(?:jpg|jpeg|png|gif|webp))', message, re.IGNORECASE)
    return urls[0] if urls else None

def query_grok_streaming(user_input: str, history: List[Dict[str, str]] = [], model: str = DEFAULT_MODEL, image_url: Union[str, None] = None) -> Generator[str, None, None]:
    """Query Grok API with streaming response support"""
    try:
        # Build message history with proper typing
        system_message = {
            "role": "system", 
            "content": "You are an expert developer proficient in multiple programming languages, with extensive experience in secure coding practices, performance optimization, and code refactoring across various paradigms (e.g., procedural, object-oriented, functional). Your goal is to review and refactor the provided code to ensure it meets the highest standards of quality, security, and efficiency, tailored to the specified or inferred language.\n\nStep-by-Step Instructions:\n1. Understand the Codebase Context: Analyze the provided code in the context of the broader codebase. Identify opportunities to leverage existing base layer components, functions, or modules instead of reinventing functionality.\n2. Security Audit: Conduct a thorough security review. Check for vulnerabilities such as injection risks, improper input validation, authentication/authorization flaws, sensitive data exposure, and resource management issues.\n3. Remove Redundancy: Identify and eliminate redundant code, including duplicated logic, unused variables, or unnecessary computations.\n4. Eliminate TODOs and Placeholders: Remove all TODO comments, FIXMEs, or incomplete sections.\n5. Replace Magic Numbers and Hardcoded Values: Replace them with named constants or enums.\n6. Optimize Performance: Replace slow algorithms with optimized alternatives. Use efficient data structures and apply language-specific optimization techniques.\n\nProvide the fully refactored code in a single, complete block, followed by a concise summary of changes made. Do not introduce new features; only refine the existing code."
        }
        
        messages = [system_message] + history

        # Handle vision input if image URL provided
        if image_url:
            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": image_url,
                            "detail": VIDEO_DETAIL_LEVEL
                        }
                    }
                ]
            }
            messages.append(user_message)
            model = VISION_MODEL
        else:
            messages.append({"role": "user", "content": user_input})

        # Make streaming API call using OpenAI client
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

def chat_function(message: str, history: List[Dict[str, str]]) -> Generator[str, None, None]:
    """Main chat function for Gradio interface"""
    image_url = extract_image_url(message)
    for partial_response in query_grok_streaming(message, history, image_url=image_url):
        yield partial_response

def overlay_videos(base_path: str, ghost_path: str, output_path: str, alpha: float = DEFAULT_ALPHA, base_start_sec: float = MIN_START_SEC, ghost_start_sec: float = MIN_START_SEC, duration_sec: Union[float, None] = None) -> tuple[Union[str, None], str]:
    """Overlay two videos with customizable parameters"""
    try:
        # Open video captures
        cap_base = cv2.VideoCapture(base_path)
        cap_ghost = cv2.VideoCapture(ghost_path)

        if not cap_base.isOpened() or not cap_ghost.isOpened():
            return None, "Error: Could not open one or both video files."

        # Set start positions
        cap_base.set(cv2.CAP_PROP_POS_MSEC, base_start_sec * 1000)
        cap_ghost.set(cv2.CAP_PROP_POS_MSEC, ghost_start_sec * 1000)

        # Get video properties from base video
        fps = cap_base.get(cv2.CAP_PROP_FPS)
        width = int(cap_base.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_base.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate maximum frames if duration specified
        max_frames = int(duration_sec * fps) if duration_sec else None

        # Setup video writer with H.264 codec
        fourcc = cv2.VideoWriter.fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            # Fallback to mp4v if H.264 not available
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        processed_frames = 0

        # Process frames
        while cap_base.isOpened() and cap_ghost.isOpened():
            if max_frames is not None and processed_frames >= max_frames:
                break

            ret_base, frame_base = cap_base.read()
            ret_ghost, frame_ghost = cap_ghost.read()

            if not ret_base or not ret_ghost:
                break

            # Resize ghost frame to match base frame if needed
            if frame_ghost.shape[:2] != (height, width):
                frame_ghost = cv2.resize(frame_ghost, (width, height))

            # Blend frames using alpha blending
            blended = cv2.addWeighted(frame_base, 1.0 - alpha, frame_ghost, alpha, 0)

            out.write(blended)
            processed_frames += 1

        # Release resources
        cap_base.release()
        cap_ghost.release()
        out.release()

        # Add audio from base video using FFmpeg
        temp_video = output_path + TEMP_VIDEO_SUFFIX
        os.rename(output_path, temp_video)
        try:
            subprocess.run([
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', base_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                output_path
            ], check=True, capture_output=True)

            return output_path, f"Successfully processed {processed_frames} frames with audio preserved."

        except subprocess.CalledProcessError as e:
            os.rename(temp_video, output_path)  # Restore original if audio fails
            return output_path, f"Video overlay completed ({processed_frames} frames), but audio processing failed: {e}"

        finally:
            if os.path.exists(temp_video):
                os.remove(temp_video)

    except Exception as e:
        return None, f"Video processing failed: {str(e)}"
    finally:
        try:
            if 'cap_base' in locals() and cap_base is not None and cap_base.isOpened():
                cap_base.release()
        except:
            pass
        try:
            if 'cap_ghost' in locals() and cap_ghost is not None and cap_ghost.isOpened():
                cap_ghost.release()
        except:
            pass
        try:
            if 'out' in locals() and out is not None and out.isOpened():
                out.release()
        except:
            pass

def process_video_overlay(base_upload: Union[str, None], ghost_upload: Union[str, None], alpha: Union[float, None], base_start: Union[float, None], ghost_start: Union[float, None], duration: Union[float, None]) -> tuple[Union[str, None], str]:
    """Process video overlay with user inputs"""
    if not base_upload or not ghost_upload:
        return None, "Please upload both base and ghost videos."

    # Handle None alpha value and ensure it's within valid range
    alpha = alpha if alpha is not None else DEFAULT_ALPHA
    if alpha < MIN_ALPHA or alpha > MAX_ALPHA:
        return None, f"Alpha value must be between {MIN_ALPHA} and {MAX_ALPHA}"

    # Ensure start times are non-negative and handle None values
    base_start = max(MIN_START_SEC, base_start if base_start is not None else MIN_START_SEC)
    ghost_start = max(MIN_START_SEC, ghost_start if ghost_start is not None else MIN_START_SEC)

    # Process duration (None means process entire video)
    duration = duration if duration and duration > 0 else None

    return overlay_videos(
        base_upload, 
        ghost_upload, 
        OUTPUT_VIDEO_PATH, 
        alpha, 
        base_start, 
        ghost_start, 
        duration
    )

# Custom CSS for stealthy dark/light themes (dark default)
CUSTOM_CSS = """
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

body, .gradio-container {
    background-color: var(--bg-color) !important;
    color: var(--text-color) !important;
    font-family: 'Arial', sans-serif;
    transition: all 0.3s ease;
}

.gradio-container {
    border: none !important;
    max-width: 1200px !important;
    margin: auto !important;
}

input, textarea, .gr-textbox input, .gr-number input {
    background-color: var(--input-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
}

button {
    background-color: var(--button-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    transition: background-color 0.2s ease;
}

button:hover {
    background-color: var(--button-hover) !important;
}

.gr-tabitem {
    background-color: var(--bg-color) !important;
    color: var(--text-color) !important;
}

.gr-markdown, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    color: var(--text-color) !important;
}

.gr-chatbot {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border-color) !important;
}

.gr-video {
    background-color: var(--input-bg) !important;
    border: 1px solid var(--border-color) !important;
}

.gr-slider input[type="range"] {
    background-color: var(--border-color) !important;
}

/* Remove any emoji or decorative elements */
.emoji, .fun-icon { 
    display: none !important; 
}

/* Tab styling */
.tab-nav button {
    background-color: var(--button-bg) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
}

.tab-nav button.selected {
    background-color: var(--accent-color) !important;
}

/* Theme toggle button styling */
.theme-toggle {
    width: 40px !important;
    height: 30px !important;
    padding: 4px !important;
    font-size: 16px !important;
    min-width: 40px !important;
    border-radius: 6px !important;
}

/* Hide Gradio branding */
.footer, .gradio-footer, footer {
    display: none !important;
}

.gr-button[title*="Built with Gradio"] {
    display: none !important;
}

a[href*="gradio.app"] {
    display: none !important;
}

.built-with, .gradio-link {
    display: none !important;
}

div[style*="built with gradio"] {
    display: none !important;
}

.gradio-container > div:last-child {
    display: none !important;
}
"""

# Create FastAPI app for health checks
app = FastAPI()

@app.get("/")
async def health_check():
    """Health check endpoint for deployment"""
    return JSONResponse(content={"status": "healthy", "service": "cipher-chat-agent"}, status_code=200)

@app.get("/health")
async def health():
    """Additional health check endpoint"""
    return JSONResponse(content={"status": "ok"}, status_code=200)

# Create Gradio interface
with gr.Blocks(
    title="Cipher",
    css=CUSTOM_CSS
) as demo:

    with gr.Row():
        with gr.Column(scale=7):
            gr.Markdown(
                """<h1 style="font-family: 'Courier New', monospace; font-weight: bold; color: var(--text-color); text-transform: uppercase; letter-spacing: 3px; margin: 0; text-shadow: 0 0 10px var(--accent-color);">CIPHER</h1>"""
            )
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=1):
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
            alpha_slider = gr.Slider(MIN_ALPHA, MAX_ALPHA, value=DEFAULT_ALPHA, label="Opacity")
            base_start = gr.Number(value=MIN_START_SEC, label="Base Start")
            ghost_start = gr.Number(value=MIN_START_SEC, label="Ghost Start")
            duration = gr.Number(value=None, label="Duration")
        process_btn = gr.Button("Process")
        output_video = gr.Video(label="Output")
        status_output = gr.Textbox(label="Status", interactive=False)

        process_btn.click(
            fn=process_video_overlay,
            inputs=[base_upload, ghost_upload, alpha_slider, base_start, ghost_start, duration],
            outputs=[output_video, status_output]
        )


# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/chat")

# Launch the application
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )