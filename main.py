# === REFACTORED CODE ===
import os
import re
import cv2
import base64
import mimetypes
import logging
import uuid
import time
import atexit
import tempfile
import shutil
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import numpy as np
import gradio as gr
from openai import OpenAI, APIError, AuthenticationError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

XAI_API_BASE_URL: str = "https://api.x.ai/v1"
DEFAULT_MODEL: str = "grok-code-fast-1"
VISION_MODEL: str = "grok-4-fast-reasoning"
XAI_API_KEY: Optional[str] = os.getenv("XAI_API_KEY")

DEFAULT_PORT: int = 5000
MIN_PORT: int = 1
MAX_PORT: int = 65535
MAX_THREADS: int = min(os.cpu_count() or 4, 8)
MAX_FILE_SIZE_GB: int = 10
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_GB * 1024**3
MAX_IMAGE_FILE_SIZE_BYTES: int = 50 * 1024 * 1024
MAX_CODE_FILE_SIZE_BYTES: int = 10 * 1024 * 1024

SUPPORTED_VIDEO_EXTENSIONS: set[str] = {'.mp4', '.mov', '.avi', '.mkv'}
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
SUPPORTED_CODE_EXTENSIONS: set[str] = {'.py', '.js', '.txt', '.cpp', '.c', '.h', '.java', '.html', '.css', '.json', '.xml', '.md', '.sh', '.bat', '.yml', '.yaml'}
DEFAULT_FRAME_SKIP: int = 5
MAX_FRAME_SKIP: int = 10
DEFAULT_RESOLUTION_SCALE: float = 0.5
ALPHA_MIN: float = 0.1
ALPHA_MAX: float = 1.0
PROGRESS_UPDATE_INTERVAL: int = 100  
MSEC_PER_SEC: int = 1000
MAX_VIDEO_DURATION_SECONDS: int = 86400
URL_MAX_LENGTH: int = 2048
QUERY_MAX_LENGTH: int = 1024
FPS_ASSUMPTION_FOR_JS: int = 30

ECC_ITERATIONS: int = 50
ECC_EPSILON: float = 1e-10
ECC_MOTION_MODEL: int = cv2.MOTION_HOMOGRAPHY
ALIGN_THRESHOLD: float = 0.6

TEMPERATURE: float = 0.7
MAX_COMPLETION_TOKENS: int = 16384

UPLOAD_DIR: str = tempfile.mkdtemp(prefix="cipher_uploads_")
atexit.register(lambda: shutil.rmtree(UPLOAD_DIR, ignore_errors=True))
ALLOWED_BASE_DIR: Path = Path(UPLOAD_DIR).resolve()

SYSTEM_PROMPT_LEARNING: str = """You are a senior software engineer and patient mentor. Your job is to help me write, understand, and improve code — not just deliver perfect code.

Follow this strict process:

1. **Clarify First**
   - Ask me to confirm: language, environment, constraints, edge cases.
   - Never assume anything.

2. **Teach As You Go**
   - Break the problem into small steps.
   - After each step:
     - Show 1 small, complete code block.
     - Explain every line in plain English.
     - Cite official docs with URLs for every API/method used.
     - Show a test case (input to expected output).
     - Highlight 1 key concept.

3. **Prove It Works**
   - Include a runnable test or assertion.
   - Explain how to verify it locally.

4. **Ask Before Moving On**
   - End with: "Does this make sense? Ready for the next step?"

5. **No Assumptions**
   - If context is missing, ask.

Goal: Help me become a better programmer. Teach, verify, collaborate."""

SYSTEM_PROMPT_POLISH: str = """You are a principal engineer at a Fortune 500 company. Your task: take the provided Python code and deliver production-grade, zero-compromise output.

Rules:
1. 1. **Clarify First**
   - Ask me to confirm: language, environment, constraints, edge cases.
   - Never assume anything.
2. **Security**: Eliminate path traversal, injection, overflow, race conditions. Use `pathlib`, `resolve()`, `is_relative_to`, safe I/O.
3. **Performance**: O(n log n) or better. Minimize allocations. Use `ThreadPoolExecutor` wisely.
4. **Clean Code**: No magic numbers, TODOs, redundancy. Named constants. Single responsibility. Type hints.
5. **Reliability**: Validate inputs. Graceful degradation. Resource cleanup. Logging.
6. **Standards**: Adhere to the **official coding standards** of the input language (e.g., PEP 8 & PEP 20 for Python), use **OpenCV** or its language-native equivalent for computer vision.

Output:
Use the input language. Explain changes by line number. Deliver clean, production-ready code only.
# === REFACTORED CODE ===
```<input language>
Collaborate first. Explain changes made. Deliver perfection."""

def create_xai_client() -> Optional[OpenAI]:
    if not XAI_API_KEY:
        logger.warning("XAI_API_KEY not found. AI functionality is limited.")
        return None
    try:
        client = OpenAI(base_url=XAI_API_BASE_URL, api_key=XAI_API_KEY)
        client.models.list()
        logger.info("API key validation successful.")
        return client
    except AuthenticationError:
        logger.error("Invalid API key provided.")
        return None
    except APIError as e:
        logger.error(f"API error during client creation: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating xAI client: {e}")
        return None

client: Optional[OpenAI] = create_xai_client()

def extract_image_url(message: str) -> Optional[str]:
    if not message or not isinstance(message, str):
        return None
    url_pattern = re.compile(
        rf'https?://[^\s<>"\'{{}}]{{1,{URL_MAX_LENGTH}}}\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s<>"\'{{}}]{{0,{QUERY_MAX_LENGTH}}})?',
        re.IGNORECASE
    )
    match = url_pattern.search(message)
    return match.group(0) if match else None

def validate_file(file_path: str, supported_extensions: set[str], max_size_bytes: int) -> bool:
    try:
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            logger.error(f"File {file_path} does not exist or is not a file.")
            return False
        if not os.access(path, os.R_OK):
            logger.error(f"File {path} is not readable.")
            return False
        if path.suffix.lower() not in supported_extensions:
            logger.error(f"Unsupported extension: {path.suffix}")
            return False
        if path.stat().st_size > max_size_bytes or path.stat().st_size <= 0:
            logger.error(f"File size invalid: {path.stat().st_size} bytes")
            return False
        if not path.is_relative_to(ALLOWED_BASE_DIR):
            logger.error(f"File {path} outside allowed directory {ALLOWED_BASE_DIR}")
            return False
        return True
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

def secure_file_upload(file_path: str) -> Optional[str]:
    """Securely move uploaded file to allowed directory to prevent path traversal."""
    try:
        source_path = Path(file_path).resolve()
        if not source_path.is_relative_to(ALLOWED_BASE_DIR):
            dest_name = uuid.uuid4().hex + source_path.suffix
            dest_path = ALLOWED_BASE_DIR / dest_name
            shutil.move(str(source_path), str(dest_path))
            return str(dest_path)
        return file_path
    except Exception as e:
        logger.error(f"File upload security error: {e}")
        return None

def query_grok_streaming(
    user_input: str,
    history: Optional[List[Tuple[str, str]]] = None,
    model: str = DEFAULT_MODEL,
    image_url: Optional[str] = None,
    file_input: Optional[str] = None,
    mode: str = "LEARNING"
) -> Generator[str, None, None]:
    if history is None:
        history = []
    if client is None:
        yield "Error: API client not available. Set XAI_API_KEY."
        return

    system_prompt = SYSTEM_PROMPT_POLISH if mode == "POLISH" else SYSTEM_PROMPT_LEARNING
    messages = [{"role": "system", "content": system_prompt}]

    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    user_content = user_input
    has_image = bool(image_url or file_input)

    if file_input:
        secure_path = secure_file_upload(file_input)
        if not secure_path:
            yield "Error: File upload security check failed."
            return
        ext = Path(secure_path).suffix.lower()
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            if not validate_file(secure_path, SUPPORTED_IMAGE_EXTENSIONS, MAX_IMAGE_FILE_SIZE_BYTES):
                yield "Error: Invalid image file."
                return
            with open(secure_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            mime_type = mimetypes.guess_type(secure_path)[0] or 'image/jpeg'
            image_url = f"data:{mime_type};base64,{image_data}"
            has_image = True
        elif ext in SUPPORTED_CODE_EXTENSIONS:
            if not validate_file(secure_path, SUPPORTED_CODE_EXTENSIONS, MAX_CODE_FILE_SIZE_BYTES):
                yield "Error: Invalid code file."
                return
            with open(secure_path, 'r', encoding='utf-8', errors='ignore') as f:
                code_content = f.read()
            user_content += "\n\nUploaded code:\n" + code_content
        else:
            yield "Error: Unsupported file type."
            return

    if has_image:
        model = VISION_MODEL
        user_content = [
            {"type": "text", "text": user_input},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]

    messages.append({"role": "user", "content": user_content})
    full_response = ""
    continuation = False

    while True:
        if continuation:
            messages.append({"role": "assistant", "content": full_response})
            messages.append({"role": "user", "content": "Continue exactly from where you left off."})
            yield "\n\n(Continuing...)\n\n"

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_COMPLETION_TOKENS,
            stream=True
        )

        finish_reason = None
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_response += delta
                yield delta
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        if finish_reason != 'length':
            break
        continuation = True
        has_image = False

def respond(
    message: str,
    chat_history: List[Tuple[str, str]],
    file_input: Optional[str],
    mode: str
) -> Generator[Tuple[List[Tuple[str, str]], str], None, None]:
    if not message.strip() and not file_input:
        yield chat_history, ""
        return
    image_url = extract_image_url(message) if message else None
    model = VISION_MODEL if (file_input and Path(file_input).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS) or image_url else DEFAULT_MODEL
    bot_message = ""
    new_history = chat_history + [(message or "[File uploaded]", bot_message)]
    yield new_history, ""  # Yield initial empty response
    # Stream the response
    for delta in query_grok_streaming(
        message or "", 
        [(h, a) for h, a in chat_history], 
        model=model, 
        image_url=image_url, 
        file_input=file_input, 
        mode=mode
    ):
        bot_message += delta
        new_history[-1] = (message or "[File uploaded]", bot_message)
        yield new_history, ""

def parse_timecode(tc: str) -> float:
    if not tc.strip():
        return 0.0
    try:
        time_part, ms_part = (tc.split('.', 1) + ['0'])[:2]
        ms = float(f"0.{ms_part.lstrip('0') or '0'}")
        parts = time_part.split(':')
        if len(parts) > 3:
            raise ValueError("Too many components")
        secs = sum(float(p) * m for p, m in zip(reversed(parts), [1, 60, 3600]))
        return secs + ms
    except Exception as e:
        raise ValueError(f"Invalid timecode: {tc}") from e

def align_frames(base_frame: np.ndarray, ghost_frame: np.ndarray, prev_warp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    ghost_gray = cv2.cvtColor(ghost_frame, cv2.COLOR_BGR2GRAY)
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            base_gray, ghost_gray, warp_matrix, ECC_MOTION_MODEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPSILON),
            inputMask=None, gaussFiltSize=5
        )
        if cc < ALIGN_THRESHOLD:
            raise cv2.error("Low correlation")
    except cv2.error:
        warp_matrix = prev_warp

    h, w = base_frame.shape[:2]
    aligned = cv2.warpPerspective(
        ghost_frame, warp_matrix, (w, h),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    return aligned, warp_matrix

def blend_frames(base: np.ndarray, aligned: np.ndarray, alpha: float) -> np.ndarray:
    return cv2.addWeighted(base, 1.0 - alpha, aligned, alpha, 0)

def overlay_videos(
    base_path: str, ghost_path: str, output_path: str,
    alpha: float, base_start_sec: float, ghost_start_sec: float,
    duration_sec: Optional[float], frame_skip: int,
    resolution_scale: float, progress: Optional[gr.Progress] = None
) -> Tuple[Optional[str], str]:
    cap_base = cap_ghost = out = None
    try:
        secure_base = secure_file_upload(base_path)
        secure_ghost = secure_file_upload(ghost_path)
        if not secure_base or not secure_ghost:
            return None, "File upload security check failed."
        if not (validate_file(secure_base, SUPPORTED_VIDEO_EXTENSIONS, MAX_FILE_SIZE_BYTES) and
                validate_file(secure_ghost, SUPPORTED_VIDEO_EXTENSIONS, MAX_FILE_SIZE_BYTES)):
            return None, "Invalid video file."

        if not (ALPHA_MIN <= alpha <= ALPHA_MAX and base_start_sec >= 0 and ghost_start_sec >= 0 and
                (duration_sec is None or duration_sec > 0) and 1 <= frame_skip <= MAX_FRAME_SKIP and
                0.1 <= resolution_scale <= 1.0):
            return None, "Invalid parameter."

        cap_base = cv2.VideoCapture(secure_base)
        cap_ghost = cv2.VideoCapture(secure_ghost)
        if not (cap_base.isOpened() and cap_ghost.isOpened()):
            return None, "Failed to open video."

        cap_base.set(cv2.CAP_PROP_POS_MSEC, base_start_sec * MSEC_PER_SEC)
        cap_ghost.set(cv2.CAP_PROP_POS_MSEC, ghost_start_sec * MSEC_PER_SEC)

        fps = cap_base.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return None, "Invalid FPS."

        width = int(cap_base.get(cv2.CAP_PROP_FRAME_WIDTH) * resolution_scale)
        height = int(cap_base.get(cv2.CAP_PROP_FRAME_HEIGHT) * resolution_scale)
        if width <= 0 or height <= 0:
            return None, "Invalid dimensions."

        base_frames_left = int(cap_base.get(cv2.CAP_PROP_FRAME_COUNT)) - int(cap_base.get(cv2.CAP_PROP_POS_FRAMES))
        ghost_frames_left = int(cap_ghost.get(cv2.CAP_PROP_FRAME_COUNT)) - int(cap_ghost.get(cv2.CAP_PROP_POS_FRAMES))
        max_frames = min(base_frames_left, ghost_frames_left,
                         int(duration_sec * fps) if duration_sec else base_frames_left)
        total_output_frames = max_frames // frame_skip

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))
        if not out.isOpened():
            return None, "Failed to initialize writer. FFmpeg required."

        processed = 0
        prev_warp = np.eye(3, 3, dtype=np.float32)

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            while processed < total_output_frames:
                batch_size = min(MAX_THREADS, total_output_frames - processed)
                batch_bases = []
                batch_ghosts = []

                for _ in range(batch_size):
                    ret_b, frame_b = cap_base.read()
                    ret_g, frame_g = cap_ghost.read()
                    if not (ret_b and ret_g):
                        break
                    if resolution_scale != 1.0:
                        frame_b = cv2.resize(frame_b, (width, height), interpolation=cv2.INTER_CUBIC)
                        frame_g = cv2.resize(frame_g, (width, height), interpolation=cv2.INTER_CUBIC)
                    batch_bases.append(frame_b)
                    batch_ghosts.append(frame_g)
                    for _ in range(frame_skip - 1):
                        cap_base.read()
                        cap_ghost.read()

                if not batch_bases:
                    break

                aligned_ghosts = []
                for b, g in zip(batch_bases, batch_ghosts):
                    aligned, prev_warp = align_frames(b, g, prev_warp)
                    aligned_ghosts.append(aligned)

                futures = [executor.submit(blend_frames, b, a, alpha) for b, a in zip(batch_bases, aligned_ghosts)]
                for f in futures:
                    out.write(f.result())

                processed += len(batch_bases)
                if progress and processed % PROGRESS_UPDATE_INTERVAL == 0:
                    progress(processed / total_output_frames, desc=f"Processing {processed}/{total_output_frames}")

        if progress:
            progress(1.0, desc="Complete")
        return output_path, f"Successfully processed {processed} frames. Saved to: {output_path}"

    except Exception as e:
        logger.error(f"Overlay error: {e}")
        return None, f"Video processing failed: {e}"
    finally:
        for cap in (cap_base, cap_ghost):
            if cap:
                cap.release()
        if out:
            out.release()

def process_video_overlay(
    base_upload: Optional[str], ghost_upload: Optional[str],
    alpha: float, base_start: str, ghost_start: str,
    duration: str, frame_skip: int, resolution_scale: float,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], Optional[str], dict, str]:
    logger.info(f"Received inputs: base_upload={base_upload}, ghost_upload={ghost_upload}, alpha={alpha}, frame_skip={frame_skip}")
    if not base_upload or not ghost_upload:
        return None, None, gr.update(visible=False), "Please upload both base and ghost videos."

    try:
        base_start_sec = parse_timecode(base_start)
        ghost_start_sec = parse_timecode(ghost_start)
        duration_sec = parse_timecode(duration) if duration else None
    except ValueError as e:
        return None, None, gr.update(visible=False), str(e)

    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]
    output_path = f"overlay_output_{timestamp}_{unique_id}.mp4"
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
    download_update = gr.update(value=result_path, visible=True) if result_path and os.path.exists(result_path) else gr.update(visible=False)
    logger.info(f"Overlay result: path={result_path}, message={status_msg}")
    return result_path, result_path, download_update, status_msg

def get_current_time_js(video_id: str) -> str:
    return f"""() => {{
        const vid = document.querySelector('#{video_id} video');
        if (!vid) return '00:00:00.000';
        const t = Math.min(vid.currentTime, {MAX_VIDEO_DURATION_SECONDS});
        const hours = String(Math.floor(t / 3600)).padStart(2, '0');
        const mins = String(Math.floor((t % 3600) / 60)).padStart(2, '0');
        const secs = String(Math.floor(t % 60)).padStart(2, '0');
        const ms = String(Math.floor((t % 1) * 1000)).padStart(3, '0');
        return `${{hours}}:${{mins}}:${{secs}}.${{ms}}`;
    }}"""

CUSTOM_CSS: str = """
meta[name="viewport"] { content: "width=device-width, initial-scale=1.0, maximum-scale=5.0"; }
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
    font-size: 16px;
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
    font-size: 16px;
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
.gr-video video:focus {
    outline: none;
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
    padding: 0;
    border-radius: 50%;
    background: linear-gradient(180deg, #ffffff 50%, #000000 50%);
    border: 1px solid var(--border-color);
    cursor: pointer;
    transition: transform 0.3s ease;
}
.theme-toggle:hover {
    transform: scale(1.1);
}
.mode-toggle {
    padding: 4px 8px;
    font-size: 12px;
    min-height: 24px;
}
.mode-toggle.hidden {
    display: none;
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
.submit-btn {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
}
.attach-btn {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    font-size: 16px;
}
@media (max-width: 768px) {
    html, body { -webkit-text-size-adjust: none; touch-action: manipulation; }
    .gradio-container { padding: 4px; }
    .gr-row { flex-direction: column; gap: 8px; }
    .gr-column { width: 100%; min-width: 0; margin-bottom: 8px; }
    input, textarea, .gr-textbox input, .gr-textbox textarea {
        font-size: 16px;
        padding: 8px;
        min-height: 40px;
        border-radius: 4px;
    }
    button, .gr-button {
        min-height: 40px;
        padding: 8px 12px;
        font-size: 16px;
    }
    .gr-chatbot, .chatbot { min-height: 70vh; }
    .gr-video, .video-container { max-width: 100%; height: auto; }
    .input-container { gap: 4px; padding: 2px 4px; }
}
@media (min-width: 769px) {
    .gradio-container { padding: 8px; }
    .gr-chatbot, .chatbot { min-height: 60vh; }
}
.emoji, .fun-icon { display: none; }
"""

with gr.Blocks(title="Cipher Code", css=CUSTOM_CSS) as demo:
    gr.HTML("""
    <script>
    document.addEventListener('DOMContentLoaded', () => {
        const videoIds = ['base_video', 'ghost_video', 'output_video'];
        videoIds.forEach(videoId => {
            const vidParent = document.querySelector(`#${videoId}`);
            if (vidParent) {
                const vid = vidParent.querySelector('video');
                if (vid) {
                    vidParent.addEventListener('keydown', (e) => {
                        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                            e.preventDefault();
                            const fps = vid.playbackRate * """ + str(FPS_ASSUMPTION_FOR_JS) + """;
                            const frameTime = 1 / fps;
                            const currentFrame = Math.round(vid.currentTime * fps);
                            const newFrame = e.key === 'ArrowRight' ? currentFrame + 1 : currentFrame - 1;
                            vid.currentTime = Math.max(0, newFrame * frameTime);
                        }
                    });
                    vidParent.addEventListener('click', () => vid.focus());
                }
            }
        });
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName.match(/^(INPUT|TEXTAREA)$/)) return;
            const key = e.key.toLowerCase();
            if (key === 'f') {
                const button = document.querySelector('#base_set_button');
                if (button) {
                    button.click();
                    e.preventDefault();
                }
            } else if (key === 'g') {
                const button = document.querySelector('#ghost_set_button');
                if (button) {
                    button.click();
                    e.preventDefault();
                }
            }
        });
    });
    </script>
    """)

    with gr.Row():
        with gr.Column(scale=9):
            gr.Markdown(
                """<h1 style="font-family: 'Courier New', monospace; font-weight: bold; color: var(--text-color); text-transform: uppercase; letter-spacing: 2px;">CIPHER Code</h1>"""
            )
        with gr.Column(scale=1, min_width=80):
            toggle_btn = gr.Button("", size="sm", elem_classes=["theme-toggle"])
            toggle_btn.click(None, js="""() => {
                document.body.classList.toggle('light');
                return null;
            }""")

    with gr.Tab("Code"):
        mode_state = gr.State("LEARNING")
        with gr.Row():
            mode_btn = gr.Button("LEARNING", variant="secondary", scale=1, elem_classes=["mode-toggle"])
        chatbot = gr.Chatbot(height="60vh")
        with gr.Row():
            # Removed file_input entirely
            with gr.Column(scale=1):
                with gr.Row(elem_classes=["input-container"]):
                    attach_btn = gr.UploadButton("📎", file_types=list(SUPPORTED_IMAGE_EXTENSIONS), elem_classes=["attach-btn"])  # Kept your original class or change to "image_upload" if preferred
                    textbox = gr.Textbox(placeholder="Toggle between learning, polish and hardcore modes at the top. Drop the full file in then start chatting", show_label=False, container=False, scale=10, lines=5)
                    submit_btn = gr.Button("Code", elem_classes=["submit-btn"])
        submit_btn.click(
            respond,
            inputs=[textbox, chatbot, attach_btn, mode_state],  # Changed file_input to attach_btn
            outputs=[chatbot, textbox]
        )

        def toggle_mode(current_mode: str):
            modes = ["LEARNING", "POLISH", "HARDCORE"]
            current_index = modes.index(current_mode)
            new_mode = modes[(current_index + 1) % len(modes)]
            return new_mode, gr.update(value=new_mode)

        mode_btn.click(
            toggle_mode,
            inputs=mode_state,
            outputs=[mode_state, mode_btn]
        )

    with gr.Tab("Video"):
        gr.Markdown(
            f"""**Note**: Maximum file size per video is {MAX_FILE_SIZE_GB}GB.
            Duration must be a positive number or empty. Frame skip (1 to {MAX_FRAME_SKIP}) and resolution scale (0.1 to 1.0) can speed up rendering.
            Start times and duration in HH:MM:SS.mmm format.
            Use left/right arrow keys to scroll videos frame-by-frame after clicking to focus.
            Press 'f' to set Base Start time or 'g' to set Ghost Start time from current position."""
        )
        with gr.Row():
            base_upload = gr.Video(label="Base Video", interactive=True, elem_id="base_video")
            ghost_upload = gr.Video(label="Ghost Video", interactive=True, elem_id="ghost_video")
        with gr.Row():
            with gr.Column():
                base_start = gr.Textbox(value="00:00:00.000", label="Base Start (HH:MM:SS.mmm)")
                set_base_start = gr.Button("Set from Current Position", elem_id="base_set_button")
            with gr.Column():
                ghost_start = gr.Textbox(value="00:00:00.000", label="Ghost Start (HH:MM:SS.mmm)")
                set_ghost_start = gr.Button("Set from Current Position", elem_id="ghost_set_button")
        with gr.Row():
            alpha_slider = gr.Slider(ALPHA_MIN, ALPHA_MAX, value=0.5, label="Opacity")
            duration = gr.Textbox(value="", label="Duration (HH:MM:SS.mmm, optional)")
            frame_skip = gr.Number(value=DEFAULT_FRAME_SKIP, label="Frame Skip", minimum=1, maximum=MAX_FRAME_SKIP, step=1, precision=0, interactive=True)
            resolution_scale = gr.Slider(0.1, 1.0, value=DEFAULT_RESOLUTION_SCALE, label="Resolution Scale")
        process_btn = gr.Button("Process")
        output_video = gr.Video(label="Output Video", elem_id="output_video")
        with gr.Row():
            save_location = gr.Textbox(label="Save Location", interactive=False)
            download_btn = gr.File(label="Download Output Video", interactive=False, visible=False)
        status_output = gr.Textbox(label="Status", interactive=False)
        set_base_start.click(fn=None, inputs=[], outputs=base_start, js=get_current_time_js("base_video"))
        set_ghost_start.click(fn=None, inputs=[], outputs=ghost_start, js=get_current_time_js("ghost_video"))
        process_btn.click(
            fn=process_video_overlay,
            inputs=[base_upload, ghost_upload, alpha_slider, base_start, ghost_start, duration, frame_skip, resolution_scale],
            outputs=[output_video, save_location, download_btn, status_output],
            show_progress="full"
        )

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", str(DEFAULT_PORT)))
        if not MIN_PORT <= port <= MAX_PORT:
            logger.warning(f"Invalid port {port}, using default {DEFAULT_PORT}")
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
            allowed_paths=[UPLOAD_DIR]
        )
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        exit(1)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        exit(1)