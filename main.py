import os
import re
import cv2
import gradio as gr
from openai import OpenAI
from typing import List, Optional, Tuple, Generator
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
XAI_API_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-code-fast-1"
VISION_MODEL = "grok-4-0709"
DEFAULT_PORT = 5000
MAX_PORT = 65535
MIN_PORT = 1
MAX_FILE_SIZE_GB = 10
MAX_THREADS = min(os.cpu_count() or 4, 20)  # Use CPU core count, capped at 20
ALPHA_MIN = 0.1
ALPHA_MAX = 1.0
MAX_TOKENS = 2000
TEMPERATURE = 0.7
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * 1024 * 1024 * 1024
DEFAULT_FRAME_SKIP = 1
DEFAULT_RESOLUTION_SCALE = 1.0  # 1.0 = original, 0.5 = half resolution
PROGRESS_UPDATE_INTERVAL = 100  # Update progress every 100 frames

# Configuration - Load API key
XAI_API_KEY = os.getenv("XAI_API_KEY")

def health_check() -> dict[str, float | str]:
    """Simple health check endpoint for deployment"""
    return {"status": "healthy", "timestamp": time.time(), "service": "grok-chat-agent"}

def create_xai_client() -> Optional[OpenAI]:
    """Create xAI client with error handling"""
    if not XAI_API_KEY:
        logger.warning("XAI_API_KEY not found. AI functionality is limited.")
        return None
    try:
        client = OpenAI(base_url=XAI_API_BASE_URL, api_key=XAI_API_KEY)
        client.models.list()
        logger.info("API key validation successful")
        return client
    except Exception as e:
        logger.error(f"Error creating xAI client: {e}")
        return None

# Create xAI client
client = create_xai_client()

def extract_image_url(message: str) -> Optional[str]:
    """Extract image URLs from user message"""
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
    history: Optional[list] = None,
    model: str = DEFAULT_MODEL,
    image_url: Optional[str] = None
) -> Generator[str, None, None]:
    """Query Grok API with streaming response support"""
    if history is None:
        history = []
    if client is None:
        yield "Error: API client not available. Please set XAI_API_KEY."
        return
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert developer proficient in multiple programming languages, "
                    "with extensive experience in secure coding practices, performance optimization, "
                    "and code refactoring. Your goal is to review and refactor provided code to meet "
                    "high standards of quality, security, and efficiency.\n\n"
                    "Steps:\n"
                    "1. Analyze codebase context and leverage existing components.\n"
                    "2. Conduct a security audit for vulnerabilities.\n"
                    "3. Remove redundant code and unused variables.\n"
                    "4. Eliminate TODOs and placeholders.\n"
                    "5. Replace magic numbers with constants.\n"
                    "6. Optimize performance with efficient algorithms.\n"
                    "Provide refactored code in a single block, followed by a summary of changes."
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
        yield f"Error communicating with Grok API: {e}"
        return

def chat_function(message: str, history: list) -> str:
    """Main chat function for Gradio interface"""
    image_url = extract_image_url(message)
    full_response = ""
    for partial_response in query_grok_streaming(message, history, image_url=image_url):
        full_response = partial_response
    return full_response

def validate_file_size(file_path: str) -> bool:
    """Validate file size before processing"""
    try:
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            logger.error(f"File {file_path} size {file_size} bytes exceeds limit of {MAX_FILE_SIZE_BYTES} bytes")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating file size for {file_path}: {e}")
        return False

def validate_file_path(file_path: str) -> bool:
    """Validate that the file path exists and is accessible"""
    try:
        if not file_path or not isinstance(file_path, str):
            return False
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist or is not accessible")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating file path {file_path}: {e}")
        return False

def process_frame(
    frame_base: np.ndarray,
    frame_ghost: np.ndarray,
    width: int,
    height: int,
    alpha: float
) -> np.ndarray:
    """Process a single frame pair for overlay"""
    if frame_ghost.shape[:2] != (height, width):
        frame_ghost = cv2.resize(frame_ghost, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.addWeighted(frame_base, 1.0 - alpha, frame_ghost, alpha, 0)

def overlay_videos(
    base_path: str,
    ghost_path: str,
    output_path: str,
    alpha: float,
    base_start_sec: float,
    ghost_start_sec: float,
    duration_sec: Optional[float],
    frame_skip: int = DEFAULT_FRAME_SKIP,
    resolution_scale: float = DEFAULT_RESOLUTION_SCALE,
    progress = None
) -> tuple[Optional[str], str]:
    """Overlay two videos with customizable parameters, progress tracking, and multithreading"""
    try:
        # Validate inputs
        if not validate_file_path(base_path) or not validate_file_path(ghost_path):
            return None, "Error: One or both video files are invalid or inaccessible."
        if not validate_file_size(base_path) or not validate_file_size(ghost_path):
            return None, f"Error: One or both video files exceed the {MAX_FILE_SIZE_GB}GB limit."
        if not ALPHA_MIN <= alpha <= ALPHA_MAX:
            return None, f"Error: Alpha must be between {ALPHA_MIN} and {ALPHA_MAX}."
        if base_start_sec < 0:
            return None, "Error: Base start time must be non-negative."
        if ghost_start_sec < 0:
            return None, "Error: Ghost start time must be non-negative."
        if duration_sec is not None and duration_sec <= 0:
            return None, "Error: Duration must be positive or empty."
        if frame_skip < 1:
            return None, "Error: Frame skip must be at least 1."
        if not 0.1 <= resolution_scale <= 1.0:
            return None, "Error: Resolution scale must be between 0.1 and 1.0."

        cap_base = cv2.VideoCapture(base_path)
        cap_ghost = cv2.VideoCapture(ghost_path)
        if not cap_base.isOpened() or not cap_ghost.isOpened():
            return None, "Error: Could not open one or both video files."

        cap_base.set(cv2.CAP_PROP_POS_MSEC, base_start_sec * 1000)
        cap_ghost.set(cv2.CAP_PROP_POS_MSEC, ghost_start_sec * 1000)
        fps = cap_base.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return None, "Error: Invalid FPS value in base video."
        width = int(cap_base.get(cv2.CAP_PROP_FRAME_WIDTH) * resolution_scale)
        height = int(cap_base.get(cv2.CAP_PROP_FRAME_HEIGHT) * resolution_scale)
        if width <= 0 or height <= 0:
            return None, "Error: Invalid video dimensions."

        # Pre-check frame compatibility
        ret_base, frame_base = cap_base.read()
        ret_ghost, frame_ghost = cap_ghost.read()
        if not ret_base or not ret_ghost:
            return None, "Error: Could not read initial frames."
        needs_resize = frame_ghost.shape[:2] != frame_base.shape[:2]
        cap_base.set(cv2.CAP_PROP_POS_MSEC, base_start_sec * 1000)  # Reset position

        total_frames = int(cap_base.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = int(duration_sec * fps) if duration_sec is not None else total_frames
        if max_frames is None or max_frames > total_frames:
            max_frames = total_frames
        max_frames = max_frames // frame_skip  # Adjust for frame skipping

        fourcc = cv2.VideoWriter.fourcc(*'H264')  # Use H264 for faster encoding
        out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))
        if not out.isOpened():
            return None, "Error: Could not initialize video writer."

        processed_frames = 0
        frame_buffer = []
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            while cap_base.isOpened() and cap_ghost.isOpened():
                if max_frames is not None and processed_frames >= max_frames:
                    break
                # Read frames with skipping
                for _ in range(frame_skip):
                    ret_base, frame_base = cap_base.read()
                    ret_ghost, frame_ghost = cap_ghost.read()
                    if not ret_base or not ret_ghost:
                        break
                if not ret_base or not ret_ghost:
                    break

                # Resize base frame if needed
                if resolution_scale != 1.0:
                    frame_base = cv2.resize(frame_base, (width, height), interpolation=cv2.INTER_AREA)

                # Process frame in parallel
                future = executor.submit(process_frame, frame_base, frame_ghost, width, height, alpha)
                frame_buffer.append(future)

                # Write frames in batches to reduce I/O overhead
                if len(frame_buffer) >= MAX_THREADS:
                    for future in frame_buffer:
                        blended = future.result()
                        out.write(blended)
                    processed_frames += len(frame_buffer)
                    if progress and max_frames > 0 and processed_frames % PROGRESS_UPDATE_INTERVAL == 0:
                        progress(processed_frames / max_frames, desc=f"Processing {processed_frames}/{max_frames} frames")
                    frame_buffer = []

        # Write remaining frames
        for future in frame_buffer:
            blended = future.result()
            out.write(blended)
        processed_frames += len(frame_buffer)
        if progress and max_frames > 0:
            progress(processed_frames / max_frames, desc=f"Processing {processed_frames}/{max_frames} frames")

        cap_base.release()
        cap_ghost.release()
        out.release()
        return output_path, f"Successfully processed {processed_frames} frames. Video saved to: {output_path}"
    except Exception as e:
        logger.error(f"Overlay error: {e}")
        return None, f"Video processing failed: {e}"
    finally:
        if 'cap_base' in locals():
            cap_base.release()
        if 'cap_ghost' in locals():
            cap_ghost.release()
        if 'out' in locals():
            out.release()

def process_video_overlay(
    base_upload: str,
    ghost_upload: str,
    alpha: float,
    base_start: float,
    ghost_start: float,
    duration: Optional[float],
    frame_skip: int,
    resolution_scale: float,
    progress = gr.Progress()
) -> tuple[Optional[str], Optional[str], str]:
    """Process video overlay with user inputs and progress tracking"""
    logger.info(f"Received inputs: base_upload={base_upload}, ghost_upload={ghost_upload}, alpha={alpha}, base_start={base_start}, ghost_start={ghost_start}, duration={duration}, frame_skip={frame_skip}, resolution_scale={resolution_scale}")

    if not base_upload or not ghost_upload:
        return None, None, "Please upload both base and ghost videos."
    if not isinstance(alpha, (int, float)) or not ALPHA_MIN <= alpha <= ALPHA_MAX:
        return None, None, f"Alpha value must be a number between {ALPHA_MIN} and {ALPHA_MAX}."
    if not isinstance(base_start, (int, float)) or base_start < 0:
        return None, None, "Base start time must be a non-negative number."
    if not isinstance(ghost_start, (int, float)) or ghost_start < 0:
        return None, None, "Ghost start time must be a non-negative number."
    if duration is not None:
        try:
            duration = float(duration)
            if duration <= 0:
                return None, None, "Duration must be a positive number or empty."
        except (TypeError, ValueError):
            return None, None, "Duration must be a valid number or empty."
    if not isinstance(frame_skip, int) or frame_skip < 1:
        return None, None, "Frame skip must be a positive integer."
    if not isinstance(resolution_scale, (int, float)) or not 0.1 <= resolution_scale <= 1.0:
        return None, None, "Resolution scale must be a number between 0.1 and 1.0."

    timestamp = int(time.time())
    output_path = f"overlay_output_{timestamp}.mp4"
    base_start = max(0.0, float(base_start))
    ghost_start = max(0.0, float(ghost_start))
    duration_sec = duration if duration is not None else None

    result_path, status_msg = overlay_videos(
        base_path=base_upload,
        ghost_path=ghost_upload,
        output_path=output_path,
        alpha=alpha,
        base_start_sec=base_start,
        ghost_start_sec=ghost_start,
        duration_sec=duration_sec,
        frame_skip=frame_skip,
        resolution_scale=resolution_scale,
        progress=progress
    )
    logger.info(f"Overlay result: path={result_path}, message={status_msg}")
    return result_path, output_path, status_msg

# Custom CSS for UI with progress bar styling
CUSTOM_CSS = """
meta[name="viewport"] { content: "width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"; }
:root {
    --primary-color: #4a4a4a;
    --bg-color: #1e1e1e;
    --text-color: #d4d4d4;
    --input-bg: #2a2a2a;
    --button-bg: #3a3a3a;
    --button-hover: #4a4a4a;
    --border-color: #3a3a3a;
    --accent-color: #666666;
    --progress-bg: #333333;
    --progress-fill: #4a90e2;
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
    --progress-bg: #e0e0e0;
    --progress-fill: #007bff;
}
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
.gr-form, .gr-box, .gr-panel, .gr-block, .gr-padded, .gr-rounded, .gradio-container > div {
    background-color: transparent;
    border: none;
    box-shadow: none;
}
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
button:hover, .gr-button:hover { background-color: var(--button-hover); }
.gr-chatbot, .chatbot {
    background-color: var(--input-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    flex: 1;
    min-height: 300px;
    overflow-y: auto;
}
.gr-chatbot .message, .chatbot .message { background-color: transparent; color: var(--text-color); }
.gr-video, .video-container {
    background-color: var(--input-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    width: 100%;
    height: auto;
}
.gr-slider input[type="range"] { background-color: transparent; accent-color: var(--accent-color); }
.gr-markdown, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { color: var(--text-color); background-color: transparent; }
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
.gr-progress {
    background-color: var(--progress-bg);
    border-radius: 4px;
    height: 20px;
    margin: 8px 0;
}
.gr-progress .progress-bar {
    background-color: var(--progress-fill);
    height: 100%;
    border-radius: 4px;
    transition: width 0.2s ease;
}
@media (max-width: 768px) {
    html, body { -webkit-text-size-adjust: none; touch-action: manipulation; }
    .gradio-container { padding: 8px; }
    .gr-row { flex-direction: column; gap: 12px; }
    .gr-column { width: 100%; min-width: 0; margin-bottom: 12px; }
    input, textarea, .gr-textbox input, .gr-textbox textarea { font-size: 16px; padding: 14px; min-height: 48px; border-radius: 8px; }
    button, .gr-button { min-height: 48px; padding: 12px 16px; font-size: 16px; }
    .gr-chatbot, .chatbot { min-height: 50vh; }
    .gr-video, .video-container { max-width: 100%; height: auto; }
}
@media (min-width: 769px) {
    .gradio-container { padding: 16px; }
    .gr-chatbot, .chatbot { min-height: 70vh; }
}
.emoji, .fun-icon { display: none; }
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
        gr.Markdown(
            f"**Note**: Maximum file size per video is {MAX_FILE_SIZE_GB}GB. "
            "Duration must be a positive number or empty. Frame skip (1 = all frames, 2 = every other frame, etc.) and resolution scale (0.1 to 1.0) can speed up rendering."
        )
        with gr.Row():
            base_upload = gr.Video(label="Base Video")
            ghost_upload = gr.Video(label="Ghost Video")
        with gr.Row():
            alpha_slider = gr.Slider(ALPHA_MIN, ALPHA_MAX, value=0.5, label="Opacity")
            base_start = gr.Number(value=0.0, label="Base Start (s)")
            ghost_start = gr.Number(value=0.0, label="Ghost Start (s)")
            duration = gr.Number(value=None, label="Duration (s)")
            frame_skip = gr.Number(value=DEFAULT_FRAME_SKIP, label="Frame Skip", minimum=1, step=1, precision=0)
            resolution_scale = gr.Slider(0.1, 1.0, value=DEFAULT_RESOLUTION_SCALE, label="Resolution Scale")
        process_btn = gr.Button("Process")
        output_video = gr.Video(label="Output Video")
        save_location = gr.Textbox(label="Save Location", interactive=False)
        status_output = gr.Textbox(label="Status", interactive=False)

        process_btn.click(
            fn=process_video_overlay,
            inputs=[base_upload, ghost_upload, alpha_slider, base_start, ghost_start, duration, frame_skip, resolution_scale],
            outputs=[output_video, save_location, status_output],
            show_progress="full"
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