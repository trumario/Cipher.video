import os
import json
import re
import cv2
import subprocess
import requests
import gradio as gr
import bcrypt
import psycopg2
from openai import OpenAI
from typing import List, Dict, Union, Any

# Configuration - Load API key from environment variables
XAI_API_KEY = os.getenv("XAI_API_KEY")
DEFAULT_MODEL = "grok-2-1212"  # Text/coding default
VISION_MODEL = "grok-4-0709"  # Vision-capable model for video and image analysis
DATABASE_URL = os.getenv("DATABASE_URL")

if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not set in environment variables.")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in environment variables.")

# Create xAI client using OpenAI pattern
client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=XAI_API_KEY
)

# Database setup
def init_db():
    """Initialize database with users table"""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(username, password):
    """Register a new user"""
    if not username or not password:
        return False, "Username and password required"
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Check if user exists
        cur.execute("SELECT username FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            return False, "Username already exists"
        
        # Create user
        password_hash = hash_password(password)
        cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s)", 
                   (username, password_hash))
        conn.commit()
        cur.close()
        conn.close()
        return True, "Registration successful"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def authenticate_user(username, password):
    """Authenticate user credentials"""
    if not username or not password:
        return False, "Username and password required"
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE username = %s", (username,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result and verify_password(password, result[0]):
            return True, "Login successful"
        return False, "Invalid credentials"
    except Exception as e:
        return False, f"Authentication failed: {str(e)}"

# Initialize database
init_db()

def extract_image_url(message):
    """Extract image URLs from user message using regex"""
    urls = re.findall(r'(https?://\S+\.(?:jpg|jpeg|png|gif|webp))', message, re.IGNORECASE)
    return urls[0] if urls else None

def query_grok_streaming(user_input, history=[], model=DEFAULT_MODEL, image_url=None):
    """Query Grok API with streaming response support"""
    try:
        # Build message history
        messages = [
            {
                "role": "system", 
                "content": "You are an expert developer proficient in multiple programming languages, with extensive experience in secure coding practices, performance optimization, and code refactoring across various paradigms (e.g., procedural, object-oriented, functional). Your goal is to review and refactor the provided code to ensure it meets the highest standards of quality, security, and efficiency, tailored to the specified or inferred language.\n\nStep-by-Step Instructions:\n1. Understand the Codebase Context: Analyze the provided code in the context of the broader codebase. Identify opportunities to leverage existing base layer components, functions, or modules instead of reinventing functionality.\n2. Security Audit: Conduct a thorough security review. Check for vulnerabilities such as injection risks, improper input validation, authentication/authorization flaws, sensitive data exposure, and resource management issues.\n3. Remove Redundancy: Identify and eliminate redundant code, including duplicated logic, unused variables, or unnecessary computations.\n4. Eliminate TODOs and Placeholders: Remove all TODO comments, FIXMEs, or incomplete sections.\n5. Replace Magic Numbers and Hardcoded Values: Replace them with named constants or enums.\n6. Optimize Performance: Replace slow algorithms with optimized alternatives. Use efficient data structures and apply language-specific optimization techniques.\n\nProvide the fully refactored code in a single, complete block, followed by a concise summary of changes made. Do not introduce new features; only refine the existing code."
            }
        ]
        
        # Add conversation history
        for human, ai in history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": ai})
        
        # Handle vision input if image URL provided
        if image_url:
            # For vision models, use proper message format
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
            model = VISION_MODEL  # Switch to vision model
        else:
            messages.append({"role": "user", "content": user_input})
        
        # Make streaming API call
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.7,
                "stream": True
            },
            stream=True
        )
        
        partial_message = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk_data = json.loads(data)
                        if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                            delta = chunk_data['choices'][0].get('delta', {})
                            if 'content' in delta and delta['content'] is not None:
                                partial_message += delta['content']
                                yield partial_message
                    except json.JSONDecodeError:
                        continue
                
    except Exception as e:
        yield f"Error communicating with Grok API: {str(e)}"

# Global authentication state (simple approach for demo)
auth_state = {"authenticated": False, "username": None}

def chat_function(message, history):
    """Main chat function for Gradio interface"""
    
    # Convert messages format to tuples format for compatibility
    history_tuples = []
    if isinstance(history, list) and len(history) > 0:
        if isinstance(history[0], dict):
            # Messages format - convert to tuples
            i = 0
            while i < len(history) - 1:
                if (history[i].get('role') == 'user' and 
                    i + 1 < len(history) and 
                    history[i + 1].get('role') == 'assistant'):
                    history_tuples.append((history[i]['content'], history[i + 1]['content']))
                    i += 2
                else:
                    i += 1
        else:
            # Already tuples format
            history_tuples = history
    
    image_url = extract_image_url(message)
    for partial_response in query_grok_streaming(message, history_tuples, image_url=image_url):
        yield partial_response

def overlay_videos(base_path, ghost_path, output_path, alpha=0.5, base_start_sec=0.0, ghost_start_sec=0.0, duration_sec=None):
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
        try:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        except:
            # Fallback to mp4v if H.264 not available
            fourcc = cv2.VideoWriter.fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
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
            if frame_ghost.shape != frame_base.shape:
                frame_ghost = cv2.resize(frame_ghost, (width, height))
            
            # Blend frames using alpha blending
            blended = cv2.addWeighted(frame_base, 1.0 - alpha, frame_ghost, alpha, 0)
            
            out.write(blended)
            processed_frames += 1
            frame_count += 1
        
        # Release resources
        cap_base.release()
        cap_ghost.release()
        out.release()
        
        # Add audio from base video using FFmpeg
        try:
            temp_video = output_path + '_temp_video_only.mp4'
            os.rename(output_path, temp_video)
            
            # Use FFmpeg to combine processed video with original audio
            subprocess.run([
                'ffmpeg', '-y',  # Overwrite output files
                '-i', temp_video,  # Video input
                '-i', base_path,   # Audio source
                '-c:v', 'copy',    # Copy video codec
                '-c:a', 'aac',     # Audio codec
                '-map', '0:v:0',   # Map video from first input
                '-map', '1:a:0',   # Map audio from second input
                '-shortest',       # Match shortest stream
                output_path
            ], check=True, capture_output=True)
            
            # Clean up temporary file
            os.remove(temp_video)
            return output_path, f"Successfully processed {processed_frames} frames with audio preserved."
            
        except subprocess.CalledProcessError as e:
            return output_path, f"Video overlay completed ({processed_frames} frames), but audio processing failed: {e}"
        except Exception as e:
            return output_path, f"Video overlay completed ({processed_frames} frames), but audio processing failed: {str(e)}"
            
    except Exception as e:
        return None, f"Video processing failed: {str(e)}"

def process_video_overlay(base_upload, ghost_upload, alpha, base_start, ghost_start, duration):
    """Process video overlay with user inputs"""
    
    if not base_upload or not ghost_upload:
        return None, "Please upload both base and ghost videos."
    
    # Handle None alpha value and ensure it's within valid range
    alpha = alpha or 0.5
    if alpha < 0.1 or alpha > 1.0:
        return None, "Alpha value must be between 0.1 and 1.0"
    
    output_path = "output_overlay.mp4"
    
    # Ensure start times are non-negative and handle None values
    base_start = max(0.0, base_start or 0.0)
    ghost_start = max(0.0, ghost_start or 0.0)
    
    # Process duration (None means process entire video)
    duration = duration if duration and duration > 0 else None
    
    return overlay_videos(
        base_upload, 
        ghost_upload, 
        output_path, 
        alpha, 
        base_start, 
        ghost_start, 
        duration
    )

def handle_login(username, password):
    """Handle user login"""
    success, message = authenticate_user(username, password)
    if success:
        auth_state['authenticated'] = True
        auth_state['username'] = username
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), message
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), message

def handle_register(username, password):
    """Handle user registration"""
    success, message = register_user(username, password)
    return message

def handle_logout():
    """Handle user logout"""
    auth_state['authenticated'] = False
    auth_state['username'] = None
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), "Logged out successfully"

def toggle_login_modal():
    """Toggle login modal visibility"""
    return gr.update(visible=True)

def close_login_modal():
    """Close login modal"""
    return gr.update(visible=False)


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

/* Login modal styling */
.login-modal {
    position: fixed !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    background-color: var(--input-bg) !important;
    border: 2px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 20px !important;
    z-index: 1000 !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
}

.login-overlay {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background-color: rgba(0,0,0,0.7) !important;
    z-index: 999 !important;
}

.login-button {
    width: 60px !important;
    height: 30px !important;
    padding: 4px 8px !important;
    font-size: 12px !important;
    min-width: 60px !important;
    border-radius: 4px !important;
}
"""

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
            alpha_slider = gr.Slider(0.1, 1.0, value=0.5, label="Opacity")
            base_start = gr.Number(value=0.0, label="Base Start")
            ghost_start = gr.Number(value=0.0, label="Ghost Start")
            duration = gr.Number(value=None, label="Duration")
        process_btn = gr.Button("Process")
        output_video = gr.Video(label="Output")
        status_output = gr.Textbox(label="Status", interactive=False)
        
        process_btn.click(
            fn=process_video_overlay,
            inputs=[base_upload, ghost_upload, alpha_slider, base_start, ghost_start, duration],
            outputs=[output_video, status_output]
        )
    

# Launch the application
if __name__ == "__main__":
    # Initialize database
    init_db()
    
    # Since Gradio serves the root path, we'll rely on the fact that
    # it returns a 200 status code for the deployment health check
    # The Gradio interface at / will serve as the health check endpoint
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
        share=False,
        show_error=True,
        quiet=False,
        show_api=False
    )
