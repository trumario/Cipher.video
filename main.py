import os
import re
import cv2
import base64
import mimetypes
from openai import OpenAI, APIError, AuthenticationError
from typing import List, Optional, Tuple, Generator
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import gradio as gr
from pathlib import Path
import uuid
from urllib.parse import quote

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
XAI_API_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-code-fast-1"
VISION_MODEL = "grok-4-fast-reasoning"
DEFAULT_PORT = 5000
MIN_PORT = 1
MAX_PORT = 65535
MAX_FILE_SIZE_GB = 10
MAX_THREADS = min(os.cpu_count() or 4, 20)
ALPHA_MIN = 0.1
ALPHA_MAX = 1.0
MAX_COMPLETION_TOKENS = 16384
TEMPERATURE = 0.7
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * 1024**3
DEFAULT_FRAME_SKIP = 1
MAX_FRAME_SKIP = 10
DEFAULT_RESOLUTION_SCALE = 1.0
PROGRESS_UPDATE_INTERVAL = 100
SUPPORTED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}
MSEC_PER_SEC = 1000
URL_MAX_LENGTH = 2048
QUERY_MAX_LENGTH = 1024
FPS_ASSUMPTION_FOR_JS = 30
MAX_VIDEO_DURATION_SECONDS = 86400  # 24 hours, reasonable upper bound for video duration
ECC_ITERATIONS = 50
ECC_EPSILON = 1e-10
ECC_MOTION_MODEL = cv2.MOTION_HOMOGRAPHY
ALIGN_THRESHOLD = 0.6
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
MAX_IMAGE_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB for images, separate from videos

# Configuration - Load API key securely
XAI_API_KEY = os.getenv("XAI_API_KEY")
if XAI_API_KEY:
    logger.info("API key loaded from environment.")
else:
    logger.warning("XAI_API_KEY not found. AI functionality is limited.")

def create_xai_client() -> Optional[OpenAI]:
    """Create xAI client with error handling and secure validation."""
    if not XAI_API_KEY:
        return None
    try:
        client = OpenAI(base_url=XAI_API_BASE_URL, api_key=XAI_API_KEY)
        client.models.list()  # Validate API key without logging sensitive data
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

# Create xAI client
client = create_xai_client()

def extract_image_url(message: str) -> Optional[str]:
    """Extract image URLs from user message with bounds checking."""
    if not message or not isinstance(message, str):
        return None
    url_pattern = re.compile(
        rf'https?://[^\s<>"\'{{}}]{{1,{URL_MAX_LENGTH}}}\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s<>"\'{{}}]{{0,{QUERY_MAX_LENGTH}}})?',
        re.IGNORECASE
    )
    match = url_pattern.search(message)
    return match.group(0) if match else None

def validate_image_file(file_path: str) -> bool:
    """Validate image file for security: existence, readability, extension, size."""
    try:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            logger.error(f"Image file {file_path} does not exist or is not a file.")
            return False
        if not os.access(file_path, os.R_OK):
            logger.error(f"Image file {file_path} is not readable.")
            return False
        ext = path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            logger.error(f"Image file {file_path} has unsupported extension {ext}.")
            return False
        file_size = path.stat().st_size
        if file_size > MAX_IMAGE_FILE_SIZE_BYTES or file_size <= 0:
            logger.error(f"Image file {file_path} size {file_size} bytes is invalid or exceeds limit.")
            return False
        # Additional security: ensure file is not in a sensitive directory
        if '..' in str(path) or str(path).startswith('/'):
            logger.error(f"Image file {file_path} has unsafe path.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating image file {file_path}: {e}")
        return False

def query_grok_streaming(
    user_input: str,
    history: Optional[List[Tuple[str, str]]] = None,
    model: str = DEFAULT_MODEL,
    image_url: Optional[str] = None,
    image_file: Optional[str] = None
) -> Generator[str, None, None]:
    """Query Grok API with streaming response support and automatic continuation if truncated."""
    if history is None:
        history = []
    if client is None:
        yield "Error: API client not available. Please set XAI_API_KEY."
        return
    try:
        system_content = (
            "You are an expert software developer with extensive experience in secure coding practices, performance optimization, and code refactoring across multiple programming languages. Your goal is to review and refactor the provided code to ensure it meets the highest standards of quality, security, and efficiency, adapting to the language and context of the input code. Additionally, evaluate and refine AI-generated code, ensuring it adheres to industry standards for efficiency, scalability, and reliability. Enhance AI-driven coding solutions, ensuring they meet enterprise-level quality and performance benchmarks.\n\n"
"Step-by-Step Instructions:\n\n"
"Understand the Codebase Context: Analyze the provided code in the context of the broader codebase. Identify opportunities to leverage existing base layer components, functions, or modules instead of reinventing functionality. Ensure the code builds upon foundational elements where appropriate, avoiding duplication.\n\n"
"Security Audit: Conduct a thorough security review. Check for vulnerabilities such as buffer overflows, integer overflows, null pointer dereferences, memory leaks, race conditions, and injection risks (e.g., SQL injection, command injection). Use secure coding patterns (e.g., bounds checking, safe string handling with strncpy/snprintf). Eliminate any insecure practices and suggest hardened alternatives.\n\n"
"Remove Redundancy: Identify and eliminate redundant code, including duplicated logic, unused variables, or unnecessary computations. Consolidate similar operations into reusable functions if they align with the codebase.\n\n"
"Eliminate TODOs and Placeholders: Remove all TODO comments, FIXMEs, or incomplete sections. Ensure the code is fully implemented and self-contained.\n\n"
"Replace Magic Numbers and Hardcoded Values: Identify any magic numbers (e.g., unexplained constants like 1024 or 5). Replace them with named constants (#define or const) that are descriptive and ideally defined in a header file or configuration module for maintainability.\n\n"
"Optimize Performance: Profile the code mentally for inefficiencies. Replace slow algorithms (e.g., O(n^2) sorts with O(n log n) or better where applicable) with optimized alternatives. Use efficient data structures, minimize allocations, and apply compiler optimizations hints if relevant. Ensure the code is performant without sacrificing readability or security.\n\n"
"General Best Practices:\n\n"
"Adhere to coding standards for the language.\n"
"Improve readability with consistent naming, indentation, and comments explaining non-obvious logic.\n"
"Add error handling for all potential failure points (e.g., check return values of malloc, fopen).\n"
"Ensure portability and avoid platform-specific assumptions unless necessary.\n"
"Make the code modular, testable, and maintainable.\n\n"
"Output Format:\n\n"
"Provide the fully refactored code in a single, complete block.\n"
"Follow the code with a concise summary of changes made, categorized by security, optimization, redundancy removal, etc.\n"
"If any assumptions were made (e.g., about the codebase), note them briefly.\n"
"Do not introduce new features; only refine the existing code.\n\n"
"Ensure the final output is perfect, secure, optimized code ready for production. If the code cannot be fully improved without additional context, state that clearly and suggest next steps."
        )
        messages = [
            {"role": "system", "content": system_content},
        ]
        for user, assistant in history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        user_content = user_input
        has_image = False
        if image_file:
            if not validate_image_file(image_file):
                yield "Error: Invalid image file."
                return
            try:
                with open(image_file, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                mime_type, _ = mimetypes.guess_type(image_file)
                if not mime_type:
                    mime_type = 'image/jpeg'  # Default fallback
                image_url = f"data:{mime_type};base64,{image_data}"
                has_image = True
            except Exception as e:
                logger.error(f"Error processing image file {image_file}: {e}")
                yield f"Error processing image: {str(e)}"
                return
        elif image_url:
            has_image = True
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
                messages.append({"role": "user", "content": "Please continue your previous response exactly from where it was cut off, without repeating or summarizing previous content."})
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
    except Exception as e:
        logger.error(f"Error querying API: {e}")
        yield f"Error querying API: {str(e)}"

def respond(
    message: str,
    chat_history: List[Tuple[str, str]],
    image_input: Optional[str]
) -> Generator[Tuple[List[Tuple[str, str]], str], None, None]:
    """Handle chat response with input validation."""
    if not message.strip() and not image_input:
        yield chat_history, ""
        return
    image_url = extract_image_url(message) if message else None
    model = VISION_MODEL if image_input or image_url else DEFAULT_MODEL
    bot_message = ""
    new_history = chat_history + [(message or "[Image uploaded]", bot_message)]
    yield new_history, ""
    for delta in query_grok_streaming(message or "", [(h, a) for h, a in chat_history], model=model, image_url=image_url, image_file=image_input):
        bot_message += delta
        new_history[-1] = (message or "[Image uploaded]", bot_message)
        yield new_history, ""

def validate_file(file_path: str, supported_extensions: set, is_video: bool = True) -> bool:
    """Validate that the file path exists, is accessible, has supported extension, and does not exceed size limit."""
    try:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            logger.error(f"File {file_path} does not exist or is not a file.")
            return False
        if not os.access(file_path, os.R_OK):
            logger.error(f"File {file_path} is not readable.")
            return False
        ext = path.suffix.lower()
        if ext not in supported_extensions:
            logger.error(f"File {file_path} has unsupported extension {ext}. Supported: {supported_extensions}")
            return False
        file_size = path.stat().st_size
        max_size = MAX_FILE_SIZE_BYTES if is_video else MAX_IMAGE_FILE_SIZE_BYTES
        if file_size > max_size or file_size <= 0:
            logger.error(f"File {file_path} size {file_size} bytes is invalid or exceeds limit of {max_size} bytes.")
            return False
        # Security: prevent directory traversal
        if '..' in str(path) or str(path).startswith('/'):
            logger.error(f"File {file_path} has unsafe path.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating file {file_path}: {e}")
        return False

def parse_timecode(tc: str) -> float:
    """Parse timecode string (HH:MM:SS.ms) to seconds as float with bounds checking."""
    if not tc.strip():
        return 0.0
    time_multipliers = [1, 60, 3600]  # seconds, minutes, hours
    component_uppers = [60, 60, None]  # upper bounds for seconds, minutes, hours (None for no upper on hours)
    try:
        if '.' in tc:
            time_part, ms_part = tc.split('.', 1)
            ms = float(f"0.{ms_part.lstrip('0') or '0'}")
        else:
            time_part = tc
            ms = 0.0
        parts = time_part.split(':')
        if len(parts) > len(time_multipliers):
            raise ValueError("Too many time components.")
        secs = 0.0
        reversed_parts = list(reversed(parts))
        for i in range(len(reversed_parts)):
            val = float(reversed_parts[i])
            if val < 0:
                raise ValueError("Negative time component value.")
            upper = component_uppers[i]
            if upper is not None and val >= upper:
                raise ValueError(f"Time component exceeds limit: {val} >= {upper} for {'seconds' if i == 0 else 'minutes'}.")
            secs += val * time_multipliers[i]
        return secs + ms
    except ValueError as e:
        raise ValueError(f"Invalid timecode format: {tc}. Use HH:MM:SS.ms") from e

def align_frames(base_frame: np.ndarray, ghost_frame: np.ndarray, prev_warp_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align ghost frame to base frame using ECC, falling back to previous warp on low correlation."""
    # Convert to grayscale for ECC
    base_gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
    ghost_gray = cv2.cvtColor(ghost_frame, cv2.COLOR_BGR2GRAY)

    # Initialize warp matrix
    warp_matrix = np.eye(3, 3, dtype=np.float32)

    try:
        cc, warp_matrix = cv2.findTransformECC(
            base_gray,
            ghost_gray,
            warp_matrix,
            ECC_MOTION_MODEL,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERATIONS, ECC_EPSILON),
            inputMask=None,
            gaussFiltSize=5
        )
        if cc < ALIGN_THRESHOLD:
            raise cv2.error("Low correlation")
    except cv2.error as e:
        logger.warning(f"ECC alignment failed (low correlation or error: {e}). Using previous warp matrix.")
        warp_matrix = prev_warp_matrix

    # Warp the ghost frame
    height, width = base_frame.shape[:2]
    aligned_ghost = cv2.warpPerspective(
        ghost_frame,
        warp_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return aligned_ghost, warp_matrix

def blend_frames(frame_base: np.ndarray, aligned_ghost: np.ndarray, alpha: float) -> np.ndarray:
    """Blend base and aligned ghost frames."""
    return cv2.addWeighted(frame_base, 1.0 - alpha, aligned_ghost, alpha, 0)

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
    """Overlay two videos with pixel-perfect alignment, progress tracking, and multithreading."""
    cap_base = None
    cap_ghost = None
    out = None
    try:
        # Validate inputs
        if not validate_file(base_path, SUPPORTED_VIDEO_EXTENSIONS, is_video=True) or not validate_file(ghost_path, SUPPORTED_VIDEO_EXTENSIONS, is_video=True):
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
        # Calculate remaining frames and total output frames for base
        base_start_frame = int(cap_base.get(cv2.CAP_PROP_POS_FRAMES))
        base_total_frames = int(cap_base.get(cv2.CAP_PROP_FRAME_COUNT))
        base_remaining_frames = base_total_frames - base_start_frame
        if base_remaining_frames <= 0:
            return None, "Error: No frames available after start time in base video."
        # Calculate for ghost to ensure sufficient frames
        ghost_start_frame = int(cap_ghost.get(cv2.CAP_PROP_POS_FRAMES))
        ghost_total_frames = int(cap_ghost.get(cv2.CAP_PROP_FRAME_COUNT))
        ghost_remaining_frames = ghost_total_frames - ghost_start_frame
        if ghost_remaining_frames <= 0:
            return None, "Error: No frames available after start time in ghost video."
        max_input_frames = min(
            base_remaining_frames,
            ghost_remaining_frames,
            int(duration_sec * fps) if duration_sec is not None else base_remaining_frames
        )
        total_output_frames = max_input_frames // frame_skip
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / frame_skip, (width, height))
        if not out.isOpened():
            return None, "Error: Could not initialize video writer. Ensure OpenCV is built with FFmpeg support."
        processed_frames = 0
        prev_warp_matrix = np.eye(3, 3, dtype=np.float32)
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            while processed_frames < total_output_frames and cap_base.isOpened() and cap_ghost.isOpened():
                batch_size = min(MAX_THREADS, total_output_frames - processed_frames)
                batch_bases = []
                batch_ghosts = []
                for _ in range(batch_size):
                    ret_base, frame_base = cap_base.read()
                    ret_ghost, frame_ghost = cap_ghost.read()
                    if not ret_base or not ret_ghost:
                        break
                    if resolution_scale != 1.0:
                        frame_base = cv2.resize(frame_base, (width, height), interpolation=cv2.INTER_CUBIC)
                        frame_ghost = cv2.resize(frame_ghost, (width, height), interpolation=cv2.INTER_CUBIC)
                    batch_bases.append(frame_base)
                    batch_ghosts.append(frame_ghost)
                    for _ in range(frame_skip - 1):
                        if not cap_base.read()[0] or not cap_ghost.read()[0]:
                            break
                batch_len = len(batch_bases)
                if batch_len == 0:
                    break
                # Sequentially align frames to maintain state
                aligned_ghosts = []
                for base_frame, ghost_frame in zip(batch_bases, batch_ghosts):
                    aligned_ghost, prev_warp_matrix = align_frames(base_frame, ghost_frame, prev_warp_matrix)
                    aligned_ghosts.append(aligned_ghost)
                # Parallelize blending
                futures = [
                    executor.submit(blend_frames, base, aligned, alpha)
                    for base, aligned in zip(batch_bases, aligned_ghosts)
                ]
                for future in futures:
                    blended = future.result()
                    out.write(blended)
                processed_frames += batch_len
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
) -> Tuple[Optional[str], Optional[str], dict, str]:
    """Process video overlay with user inputs and progress tracking."""
    logger.info(f"Received inputs: base_upload={base_upload}, ghost_upload={ghost_upload}, alpha={alpha}, base_start={base_start}, ghost_start={ghost_start}, duration={duration}, frame_skip={frame_skip}, resolution_scale={resolution_scale}")
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
    logger.info(f"Overlay result: path={result_path}, message={status_msg}, download_visible={download_update['visible']}")
    return result_path, result_path, download_update, status_msg

def get_current_time_js(video_id: str) -> str:
    """Generate JS to get current video time, format as HH:MM:SS.mmm with bounds checking."""
    return f"""() => {{
        const vid = document.querySelector('#{video_id} video');
        if (!vid) return '00:00:00.000';
        const t = vid.currentTime;
        if (t < 0 || t > {MAX_VIDEO_DURATION_SECONDS}) return '00:00:00.000';
        const hours = Math.floor(t / 3600).toString().padStart(2, '0');
        const mins = Math.floor((t % 3600) / 60).toString().padStart(2, '0');
        const secs = Math.floor(t % 60).toString().padStart(2, '0');
        const ms = Math.floor((t % 1) * 1000).toString().padStart(3, '0');
        return `${{hours}}:${{mins}}:${{secs}}.${{ms}}`;
    }}"""

# Custom CSS for minimalistic UI
CUSTOM_CSS = """
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
    cursor: pointer;
    background-color: var(--button-bg);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    transition: background-color 0.2s ease;
}
.attach-btn:hover { background-color: var(--button-hover); }
#image_upload { display: none; }
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

# Create Gradio interface
with gr.Blocks(title="Cipher", css=CUSTOM_CSS) as demo:
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
                            const fps = vid.playbackRate * """ + str(FPS_ASSUMPTION_FOR_JS) + """; // Assume 30 FPS if unknown
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
            attach_html = gr.HTML('<label for="image_upload" class="attach-btn">📎</label>')
            textbox = gr.Textbox(placeholder="Enter code or image URL...", show_label=False, container=False, scale=10, lines=5)
            submit_btn = gr.Button("↑", elem_classes=["submit-btn"])
        image_input = gr.File(label="", file_types=["image"], elem_id="image_upload")
        submit_btn.click(
            respond,
            inputs=[textbox, chatbot, image_input],
            outputs=[chatbot, textbox]
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
            frame_skip = gr.Number(value=DEFAULT_FRAME_SKIP, label="Frame Skip", minimum=1, maximum=MAX_FRAME_SKIP, step=1, precision=0)
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
            allowed_paths=[os.getcwd()]
        )
    except ValueError as ve:
        logger.error(f"Configuration error: {ve}")
        exit(1)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        exit(1)