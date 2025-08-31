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
VISION_MODEL = "grok-2-vision-1212"  # Vision-capable model

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
                "content": "You are a helpful assistant specializing in coding assistance and data analysis. For coding questions, provide clear explanations and working code examples. For images like charts or graphs, analyze the visual data in detail."
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

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

.chat-message {
    font-size: 14px !important;
}

.tab-nav {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
}

.video-upload {
    border: 2px dashed #ccc !important;
    border-radius: 10px !important;
}

.status-box {
    background-color: #f8f9fa !important;
    border-left: 4px solid #007bff !important;
    padding: 10px !important;
}
"""

# Create Gradio interface
with gr.Blocks(
    title="Grok-Powered Coding & Media Agent",
    css=custom_css,
    theme=gr.themes.Soft()
) as demo:
    
    gr.Markdown(
        """
        # 🤖 Grok-Powered Coding & Media Agent
        
        **Powered by xAI's Grok** - Your intelligent assistant for coding help and media processing.
        
        - 💬 **Chat Tab**: Get coding assistance and analyze images/charts from URLs
        - 🎥 **Video Overlay Tab**: Blend two videos with customizable effects
        """
    )
    
    with gr.Tab("💬 Chat Agent"):
        gr.Markdown("### Coding Assistant & Image Analysis")
        
        chat_interface = gr.ChatInterface(
            chat_function,
            textbox=gr.Textbox(
                placeholder="Ask about coding, paste an image URL for analysis (e.g., https://example.com/chart.png), or request help with any programming topic...",
                container=False,
                scale=7
            ),
            title="Grok Chat - Streaming Responses",
            description="🔥 **Features**: Coding assistance with grok-2-1212 | Image analysis with grok-2-vision-1212 (auto-switching)",
            examples=[
                "How do I sort a list in Python using different methods?",
                "Explain the difference between let, const, and var in JavaScript",
                "Analyze this stock chart: https://example.com/stock-chart.png",
                "Show me how to implement a binary search algorithm",
                "What are the best practices for React component design?"
            ]
        )
    
    with gr.Tab("🎥 Video Overlay"):
        gr.Markdown("### Video Overlay & Ghosting Tool")
        gr.Markdown("Upload two videos to create a blended overlay effect with customizable opacity and timing.")
        
        with gr.Row():
            with gr.Column():
                base_upload = gr.Video(
                    label="📹 Base Video",
                    elem_classes=["video-upload"]
                )
                base_start = gr.Number(
                    value=0.0,
                    label="Base Start Time (seconds)",
                    minimum=0.0,
                    step=0.1
                )
            
            with gr.Column():
                ghost_upload = gr.Video(
                    label="👻 Ghost Video (Overlay)",
                    elem_classes=["video-upload"]
                )
                ghost_start = gr.Number(
                    value=0.0,
                    label="Ghost Start Time (seconds)",
                    minimum=0.0,
                    step=0.1
                )
        
        with gr.Row():
            alpha_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="👻 Ghost Opacity (0.1 = very transparent, 1.0 = opaque)",
                info="Higher values make the ghost video more visible"
            )
            duration = gr.Number(
                value=None,
                label="Duration Limit (seconds, optional)",
                minimum=1,
                step=1,
                info="Leave empty to process entire video"
            )
        
        process_btn = gr.Button(
            "🎬 Generate Overlaid Video",
            variant="primary",
            size="lg"
        )
        
        with gr.Row():
            output_video = gr.Video(
                label="📤 Output Video",
                show_download_button=True
            )
            status_output = gr.Textbox(
                label="📊 Processing Status",
                elem_classes=["status-box"],
                interactive=False
            )
        
        # Connect the processing function
        process_btn.click(
            fn=process_video_overlay,
            inputs=[
                base_upload,
                ghost_upload,
                alpha_slider,
                base_start,
                ghost_start,
                duration
            ],
            outputs=[output_video, status_output],
            show_progress=True
        )
        
        gr.Markdown(
            """
            ### 📋 Instructions:
            1. **Upload Videos**: Select your base video and the ghost/overlay video
            2. **Adjust Timing**: Set start times for each video (useful for synchronization)
            3. **Set Opacity**: Control how transparent/opaque the ghost video appears
            4. **Optional Duration**: Limit processing time or leave empty for full video
            5. **Process**: Click generate to create your blended video with preserved audio
            
            **Supported formats**: MP4, AVI, MOV, and most common video formats
            """
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
