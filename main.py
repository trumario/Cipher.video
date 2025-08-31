import os
import json
import re
import cv2
import subprocess
import requests
import gradio as gr
from openai import OpenAI

# Configuration - Load API key from environment variables
XAI_API_KEY = os.getenv("XAI_API_KEY", "your_xai_api_key_here")
DEFAULT_MODEL = "grok-2-1212"  # Text/coding default
VISION_MODEL = "grok-4-0709"  # Vision-capable model for video and image analysis

# Create xAI client using OpenAI pattern
client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=XAI_API_KEY
)

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
            content = [
                {"type": "text", "text": user_input},
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": image_url,
                        "detail": "high"
                    }
                }
            ]
            messages.append({"role": "user", "content": content})
            model = VISION_MODEL  # Switch to vision model
        else:
            messages.append({"role": "user", "content": user_input})
        
        # Make streaming API call
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.7,
            stream=True
        )
        
        partial_message = ""
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                partial_message += chunk.choices[0].delta.content
                yield partial_message
                
    except Exception as e:
        yield f"Error communicating with Grok API: {str(e)}"

def chat_function(message, history):
    """Main chat function for Gradio interface"""
    image_url = extract_image_url(message)
    for partial_response in query_grok_streaming(message, history, image_url=image_url):
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
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        try:
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        except:
            # Fallback to mp4v if H.264 not available
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
    
    if alpha < 0.1 or alpha > 1.0:
        return None, "Alpha value must be between 0.1 and 1.0"
    
    output_path = "output_overlay.mp4"
    
    # Ensure start times are non-negative
    base_start = max(0.0, base_start)
    ghost_start = max(0.0, ghost_start)
    
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
"""

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
    demo.launch(
        server_name="0.0.0.0",
        server_port=5000,
        share=False,
        show_error=True,
        quiet=False
    )
