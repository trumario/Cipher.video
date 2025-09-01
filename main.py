import os
import re
import cv2
from openai import OpenAI
from typing import List, Optional, Tuple, Generator
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gradio as gr
import base64

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
XAI_API_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-code-fast-1"
VISION_MODEL = "grok-4-0709"
DEFAULT_PORT = 5000
MIN_PORT = 1
MAX_PORT = 65535
MAX_FILE_SIZE_GB = 10
MAX_THREADS = min(os.cpu_count() or 4, 20)
ALPHA_MIN = 0.1
ALPHA_MAX = 1.0
MAX_TOKENS = 2000
TEMPERATURE = 0.7
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * 1024**3
DEFAULT_FRAME_SKIP = 1
MAX_FRAME_SKIP = 10
DEFAULT_RESOLUTION_SCALE = 1.0
PROGRESS_UPDATE_INTERVAL = 100
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
MSEC_PER_SEC = 1000
MODELS = ["grok-code-fast-1", "grok-4-0709"]
URL_MAX_LENGTH = 2048
QUERY_MAX_LENGTH = 1024
TIME_MULTIPLIERS = [1, 60, 3600]

# Configuration - Load API key
XAI_API_KEY = os.getenv("XAI_API_KEY")

def create_xai_client() -> Optional[OpenAI]:
    """Create xAI client with error handling"""
    if not XAI_API_KEY:
        logger.warning("XAI_API_KEY not found. AI functionality is limited.")
        return None
    try:
        client = OpenAI(base_url=XAI_API_BASE_URL, api_key=XAI_API_KEY)
        client.models.list()  # Validate API key
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
        rf'https?://[^\s<>"\'{{}}]{{1,{URL_MAX_LENGTH}}}\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s<>"\'{{}}]{{0,{QUERY_MAX_LENGTH}}})?',
        re.IGNORECASE
    )
    match = url_pattern.search(message)
    return match.group(0) if match else None

def query_grok_streaming(
    user_input: str,
    history: Optional[List[Tuple[str, str]]] = None,
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
                    "You are an expert software developer with extensive experience in secure coding practices, performance optimization, and code refactoring across multiple programming languages. Your goal is to review and refactor the provided code to ensure it meets the highest standards of quality, security, and efficiency, adapting to the language and context of the input code.\n\n"
                    "Input Code:\n\n"
                    "[Input code here]\n\n"
                    "Step-by-Step Instructions:\n\n"
                    "Understand the Codebase Context: Analyze the provided code in the context of the broader codebase. Identify opportunities to leverage existing base layer components, functions, or modules instead of reinventing functionality. Ensure the code builds upon foundational elements where appropriate, avoiding duplication.\n\n"
                    "Security Audit: Conduct a thorough security review. Check for vulnerabilities such as overflows, null references, memory leaks, race conditions, and injection risks (e.g., SQL injection, command injection). Use secure coding patterns appropriate to the language (e.g., bounds checking, safe input handling). Eliminate any insecure practices and suggest hardened alternatives.\n\n"
                    "Remove Redundancy: Identify and eliminate redundant code, including duplicated logic, unused variables, or unnecessary computations. Consolidate similar operations into reusable functions if they align with the codebase.\n\n"
                    "Eliminate TODOs and Placeholders: Remove all TODO comments, FIXMEs, or incomplete sections. Ensure the code is fully implemented and self-contained.\n\n"
                    "Replace Magic Numbers and Hardcoded Values: Identify any magic numbers (e.g., unexplained constants like 1024 or 5). Replace them with named constants that are descriptive and ideally defined in a configuration section or module for maintainability, using language-appropriate mechanisms (e.g., const, enums, or defines).\n\n"
                    "Optimize Performance: Profile the code mentally for inefficiencies. Replace slow algorithms (e.g., O(n^2) sorts with O(n log n) or better where applicable) with optimized alternatives. Use efficient data structures, minimize allocations, and apply optimization techniques if relevant. Ensure the code is performant without sacrificing readability or security.\n\n"
                    "General Best Practices:\n\n"
                    "Adhere to the standards and conventions of the programming language used in the input code.\n"
                    "Improve readability with consistent naming, indentation, and comments explaining non-obvious logic.\n"
                    "Add error handling for all potential failure points (e.g., check return values of allocation or I/O functions).\n\n"
                    "Ensure portability and avoid platform-specific assumptions unless necessary.\n"
                    "Make the code modular, testable, and maintainable.\n\n"
                    "Output Format:\n\n"
                    "Provide the fully refactored code in a single, complete block.\n"
                    "Follow the code with a concise summary of changes made, categorized by security, optimization, redundancy removal, etc.\n"
                    "If any assumptions were made (e.g., about the codebase or language specifics), note them briefly.\n"
                    "Do not introduce new features; only refine the existing code.\n\n"
                    "Ensure the final output is perfect, secure, optimized code ready for production. If the code cannot be fully improved without additional context, state that clearly and suggest next steps."
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

def respond(
    message: str,
    chat_history: List[Tuple[str, str]],
    model: str,
    file_path: Optional[str]
) -> Generator[Tuple[List[Tuple[str, str]], str, Optional[str]], None, None]:
    """Handle chat response with file upload support"""
    image_url = None
    if file_path:
        if not validate_file(file_path, SUPPORTED_IMAGE_EXTENSIONS, is_video=False):
            chat_history.append((message, f"Error: Invalid file or exceeds the {MAX_FILE_SIZE_GB}GB limit."))
            yield chat_history, message, file_path
            return
        try:
            ext = os.path.splitext(file_path)[1].lower()
            with open(file_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            image_url = f"data:image/{ext[1:]};base64,{b64}"
        except Exception as e:
            chat_history.append((message, f"Error processing file: {str(e)}"))
            yield chat_history, message, file_path
            return

    if not image_url:
        image_url = extract_image_url(message)

    bot_message = ""
    new_history = chat_history + [(message, bot_message)]
    yield new_history, "", None

    for partial in query_grok_streaming(message, [(h, a) for h, a in chat_history], model=model, image_url=image_url):
        bot_message = partial
        new_history[-1] = (message, bot_message)
        yield new_history, "", None

def validate_file(file_path: str, supported_extensions: set, is_video: bool = True) -> bool:
    """Validate that the file path exists, is accessible, has supported extension, and does not exceed size limit"""
    try:
        if not file_path or not isinstance(file_path, str):
            return False
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} does not exist or is not accessible")
            return False
        if not os.access(file_path, os.R_OK):
            logger.error(f"File {file_path} is not readable")
            return False
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in supported_extensions:
            logger.error(f"File {file_path} has unsupported extension {ext}. Supported: {supported_extensions}")
            return False
        file_size = os.path.getsize(file_path)
        if file_size > MAX_FILE_SIZE_BYTES:
            logger.error(f"File {file_path} size {file_size} bytes exceeds limit of {MAX_FILE_SIZE_BYTES} bytes")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating file {file_path}: {e}")
        return False

def parse_timecode(tc: str) -> float:
    """Parse timecode string (HH:MM:SS.ms) to seconds as float"""
    if not tc.strip():
        return 0.0
    try:
        if '.' in tc:
            time_part, ms_part = tc.split('.', 1)
            ms = float(f"0.{ms_part.lstrip('0') or '0'}")
        else:
            time_part = tc
            ms = 0.0
        parts = time_part.split(':')
        secs = 0.0
        for i, part in enumerate(reversed(parts[:len(TIME_MULTIPLIERS)])):
            secs += float(part) * TIME_MULTIPLIERS[i]
        return secs + ms
    except ValueError as e:
        raise ValueError(f"Invalid timecode format: {tc}. Use HH:MM:SS.ms") from e

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
    progress: Optional[gr.Progress] = None
) -> Tuple[Optional[str], str]:
    """Overlay two videos with customizable parameters, progress tracking, and multithreading"""
    cap_base = None
    cap_ghost = None
    out = None
    try:
        # Validate inputs
        if not validate_file(base_path, SUPPORTED_VIDEO_EXTENSIONS) or not validate_file(ghost_path, SUPPORTED_VIDEO_EXTENSIONS):
            return None, "Error: One or both video files are invalid, inaccessible, have unsupported formats, or exceed size limit."
        if not ALPHA_MIN <= alpha <= ALPHA_MAX:
            return None, f"Error: Alpha must be between {ALPHA_MIN} and {ALPHA_MAX}."
        if base_start_sec < 0:
            return None, "Error: Base start time must be non-negative."
        if ghost_start_sec < 0:
            return None, "Error: Ghost start time must be non-negative."
        if duration_sec is not None and duration_sec <= 0:
            return None, "Error: Duration must be positive or empty."
        if frame_skip < 1 or frame_skip > MAX_FRAME_SKIP:
            return None, f"Error: Frame skip must be between 1 and {MAX_FRAME_SKIP}."
        if not 0.1 <= resolution_scale <= 1.0:
            return None, "Error: Resolution scale must be between 0.1 and 1.0."

        cap_base = cv2.VideoCapture(base_path)
        cap_ghost = cv2.VideoCapture(ghost_path)
        if not cap_base.isOpened() or not cap_ghost.isOpened():
            return None, "Error: Could not open one or both video files."

        cap_base.set(cv2.CAP_PROP_POS_MSEC, base_start_sec * MSEC_PER_SEC)
        cap_ghost.set(cv2.CAP_PROP_POS_MSEC, ghost_start_sec * MSEC_PER_SEC)

        fps = cap_base.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return None, "Error: Invalid FPS value in base video."

        width = int(cap_base.get(cv2.CAP_PROP_FRAME_WIDTH) * resolution_scale)
        height = int(cap_base.get(cv2.CAP_PROP_FRAME_HEIGHT) * resolution_scale)
        if width <= 0 or height <= 0:
            return None, "Error: Invalid video dimensions."

        # Calculate remaining frames and total output frames
        start_frame = int(cap_base.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(cap_base.get(cv2.CAP_PROP_FRAME_COUNT))
        remaining_frames = total_frames - start_frame
        if remaining_frames <= 0:
            return None, "Error: No frames available after start time."

        max_input_frames = int(duration_sec * fps) if duration_sec is not None else remaining_frames
        max_input_frames = min(max_input_frames, remaining_frames)
        total_output_frames = max_input_frames // frame_skip

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))
        if not out.isOpened():
            return None, "Error: Could not initialize video writer. Ensure OpenCV is built with FFmpeg support."

        processed_frames = 0
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            while processed_frames < total_output_frames and cap_base.isOpened() and cap_ghost.isOpened():
                batch_size = min(MAX_THREADS, total_output_frames - processed_frames)
                batch_frames = []
                for _ in range(batch_size):
                    ret_base, frame_base = cap_base.read()
                    ret_ghost, frame_ghost = cap_ghost.read()
                    if not ret_base or not ret_ghost:
                        break
                    if resolution_scale != 1.0:
                        frame_base = cv2.resize(frame_base, (width, height), interpolation=cv2.INTER_AREA)
                    batch_frames.append((frame_base, frame_ghost))
                    for _ in range(frame_skip - 1):
                        if not cap_base.read()[0] or not cap_ghost.read()[0]:
                            break
                if not batch_frames:
                    break

                futures = [executor.submit(process_frame, fb, fg, width, height, alpha) for fb, fg in batch_frames]
                for future in futures:
                    blended = future.result()
                    out.write(blended)

                processed_frames += len(batch_frames)
                if progress and total_output_frames > 0 and processed_frames % PROGRESS_UPDATE_INTERVAL == 0:
                    progress(processed_frames / total_output_frames, desc=f"Processing {processed_frames}/{total_output_frames} frames")

        if progress and total_output_frames > 0:
            progress(1.0, desc=f"Completed {processed_frames}/{total_output_frames} frames")

        return output_path, f"Successfully processed {processed_frames} frames. Video saved to: {output_path}"
    except Exception as e:
        logger.error(f"Overlay error: {e}")
        return None, f"Video processing failed: {e}"
    finally:
        if cap_base:
            cap_base.release()
        if cap_ghost:
            cap_ghost.release()
        if out:
            out.release()

def process_video_overlay(
    base_upload: Optional[str],
    ghost_upload: Optional[str],
    alpha: float,
    base_start: str,
    ghost_start: str,
    duration: str,
    frame_skip: int,
    resolution_scale: float,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], Optional[str], str]:
    """Process video overlay with user inputs and progress tracking"""
    logger.info(f"Received inputs: base_upload={base_upload}, ghost_upload={ghost_upload}, alpha={alpha}, base_start={base_start}, ghost_start={ghost_start}, duration={duration}, frame_skip={frame_skip}, resolution_scale={resolution_scale}")

    if not base_upload or not ghost_upload:
        return None, None, "Please upload both base and ghost videos."

    try:
        base_start_sec = parse_timecode(base_start)
        ghost_start_sec = parse_timecode(ghost_start)
        duration_sec = parse_timecode(duration) if duration else None
    except ValueError as e:
        return None, None, str(e)

    timestamp = int(time.time())
    output_path = f"overlay_output_{timestamp}.mp4"

    result_path, status_msg = overlay_videos(
        base_path=base_upload,
        ghost_path=ghost_upload,
        output_path=output_path,
        alpha=alpha,
        base_start_sec=base_start_sec,
        ghost_start_sec=ghost_start_sec,
        duration_sec=duration_sec,
        frame_skip=frame_skip,
        resolution_scale=resolution_scale,
        progress=progress
    )
    logger.info(f"Overlay result: path={result_path}, message={status_msg}")
    return result_path, output_path, status_msg

def get_current_time_js(video_id: str) -> str:
    """Generate JS to get current video time and format as HH:MM:SS.mmm"""
    return f"""() => {{
        const vid = document.querySelector('#{video_id} video');
        if (vid) {{
            const t = vid.currentTime;
            const hours = Math.floor(t / 3600).toString().padStart(2, '0');
            const mins = Math.floor((t % 3600) / 60).toString().padStart(2, '0');
            const secs = Math.floor(t % 60).toString().padStart(2, '0');
            const ms = Math.floor((t % 1) * 1000).toString().padStart(3, '0');
            return `${{hours}}:${{mins}}:${{secs}}.${{ms}}`;
        }}
        return '00:00:00.000';
    }}"""

# Custom CSS for minimalistic UI
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
    border-radius: 4px;
    font-size: 14px;
    padding: 8px;
    box-sizing: border-box;
    width: 100%;
}
button, .gr-button {
    background-color: var(--button-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 14px;
    cursor: pointer;
    transition: background-color 0.2s ease;
}
button:hover, .gr-button:hover { background-color: var(--button-hover); }
.gr-chatbot, .chatbot {
    background-color: var(--input-bg);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    flex: 1;
    min-height: 200px;
    max-height: 70vh;
    overflow-y: auto;
}
.gr-chatbot .message, .chatbot .message { 
    background-color: transparent; 
    color: var(--text-color); 
    padding: 8px;
    margin: 4px;
}
.gr-video, .video-container {
    background-color: var(--input-bg);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    width: 100%;
    height: auto;
}
.gr-slider input[type="range"] { 
    background-color: transparent; 
    accent-color: var(--accent-color); 
}
.gr-markdown, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 { 
    color: var(--text-color); 
    background-color: transparent; 
}
.theme-toggle {
    width: 32px;
    height: 32px;
    padding: 6px;
    font-size: 16px;
    border-radius: 4px;
    background-color: var(--button-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
}
.gr-progress {
    background-color: var(--progress-bg);
    border-radius: 4px;
    height: 16px;
    margin: 4px 0;
}
.gr-progress .progress-bar {
    background-color: var(--progress-fill);
    height: 100%;
    border-radius: 4px;
    transition: width 0.2s ease;
}
.input-container {
    display: flex;
    align-items: center;
    gap: 8px;
    background-color: var(--input-bg);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 4px 8px;
}
.model-select {
    background-color: var(--input-bg);
    color: var(--text-color);
    border: none;
    font-size: 14px;
    padding: 4px;
    cursor: pointer;
}
.model-select:focus {
    outline: none;
}
.file-upload-btn {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
}
.file-upload-btn input[type="file"] {
    display: none;
}
.submit-btn {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
}
@media (max-width: 768px) {
    html, body { -webkit-text-size-adjust: none; touch-action: manipulation; }
    .gradio-container { padding: 4px; }
    .gr-row { flex-direction: column; gap: 8px; }
    .gr-column { width: 100%; min-width: 0; margin-bottom: 8px; }
    input, textarea, .gr-textbox input, .gr-textbox textarea { 
        font-size: 14px; 
        padding: 8px; 
        min-height: 40px; 
        border-radius: 4px; 
    }
    button, .gr-button { 
        min-height: 40px; 
        padding: 8px 12px; 
        font-size: 14px; 
    }
    .gr-chatbot, .chatbot { min-height: 50vh; }
    .gr-video, .video-container { max-width: 100%; height: auto; }
    .input-container { gap: 4px; padding: 2px 4px; }
}
@media (min-width: 769px) {
    .gradio-container { padding: 8px; }
    .gr-chatbot, .chatbot { min-height: 60vh; }
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
        chatbot = gr.Chatbot(height="60vh")
        with gr.Row(elem_classes=["input-container"]):
            model_dropdown = gr.Dropdown(choices=MODELS, value=DEFAULT_MODEL, label=None, elem_classes=["model-select"], container=False)
            file_upload = gr.File(label=None, file_types=list(SUPPORTED_IMAGE_EXTENSIONS), elem_classes=["file-upload-btn"], container=False)
            textbox = gr.Textbox(placeholder="Enter code or image URL...", show_label=False, container=False, scale=8)
            submit_btn = gr.Button("↑", elem_classes=["submit-btn"])
        submit_btn.click(
            respond,
            inputs=[textbox, chatbot, model_dropdown, file_upload],
            outputs=[chatbot, textbox, file_upload]
        )
        textbox.submit(
            respond,
            inputs=[textbox, chatbot, model_dropdown, file_upload],
            outputs=[chatbot, textbox, file_upload]
        )

    with gr.Tab("Video"):
        gr.Markdown(
            f"**Note**: Maximum file size per video is {MAX_FILE_SIZE_GB}GB. "
            f"Duration must be a positive number or empty. Frame skip (1 to {MAX_FRAME_SKIP}) and resolution scale (0.1 to 1.0) can speed up rendering. "
            "Start times and duration in HH:MM:SS.mmm format."
        )
        with gr.Row():
            base_upload = gr.Video(label="Base Video", interactive=True, elem_id="base_video")
            ghost_upload = gr.Video(label="Ghost Video", interactive=True, elem_id="ghost_video")
        with gr.Row():
            with gr.Column():
                base_start = gr.Textbox(value="00:00:00.000", label="Base Start (HH:MM:SS.mmm)")
                set_base_start = gr.Button("Set from Current Position")
            with gr.Column():
                ghost_start = gr.Textbox(value="00:00:00.000", label="Ghost Start (HH:MM:SS.mmm)")
                set_ghost_start = gr.Button("Set from Current Position")
        with gr.Row():
            alpha_slider = gr.Slider(ALPHA_MIN, ALPHA_MAX, value=0.5, label="Opacity")
            duration = gr.Textbox(value="", label="Duration (HH:MM:SS.mmm, optional)")
            frame_skip = gr.Number(value=DEFAULT_FRAME_SKIP, label="Frame Skip", minimum=1, maximum=MAX_FRAME_SKIP, step=1, precision=0)
            resolution_scale = gr.Slider(0.1, 1.0, value=DEFAULT_RESOLUTION_SCALE, label="Resolution Scale")
        process_btn = gr.Button("Process")
        output_video = gr.Video(label="Output Video")
        save_location = gr.Textbox(label="Save Location", interactive=False)
        status_output = gr.Textbox(label="Status", interactive=False)

        set_base_start.click(fn=None, inputs=[], outputs=base_start, js=get_current_time_js("base_video"))
        set_ghost_start.click(fn=None, inputs=[], outputs=ghost_start, js=get_current_time_js("ghost_video"))
        process_btn.click(
            fn=process_video_overlay,
            inputs=[base_upload, ghost_upload, alpha_slider, base_start, ghost_start, duration, frame_skip, resolution_scale],
            outputs=[output_video, save_location, status_output],
            show_progress="full"
        )

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", str(DEFAULT_PORT)))
        if not MIN_PORT <= port <= MAX_PORT:
            logger.warning(f"Invalid port {port}, using default port {DEFAULT_PORT}")
            port = DEFAULT_PORT

        if not XAI_API_KEY:
            logger.warning("XAI_API_KEY not properly configured. AI functionality will be limited.")

        logger.info(f"Starting Gradio app on port {port}")
        logger.info("API client status: %s", "Connected" if client else "Not connected")

        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=False,
            show_error=True,
            max_threads=MAX_THREADS,
            ssl_verify=False,
            inbrowser=False,
            allowed_paths=[os.getcwd()]  # Restrict to current directory to prevent path traversal
        )
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        exit(1)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        exit(1)