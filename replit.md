# Overview

This is a Grok-Powered Coding & Media Chat Agent built with Python and Gradio. The application creates a web-based chat interface that integrates with xAI's Grok API to provide coding assistance and image analysis capabilities. Users can interact with the AI through text or by providing image URLs (particularly useful for analyzing stock charts and other visual content). The application features a clean, modern web interface with streaming responses and supports both text-based coding queries and vision-based image analysis.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **UI Framework**: Gradio-based web interface providing a chat-like experience
- **Styling**: Custom CSS with gradient themes and modern responsive design
- **User Input**: Supports both text input and image URL extraction via regex patterns
- **Response Handling**: Real-time streaming responses from the AI model

## Backend Architecture
- **Main Framework**: Python-based application using the OpenAI client pattern
- **API Integration**: Direct HTTP requests to xAI's API endpoints using the OpenAI-compatible interface
- **Model Selection**: Dynamic model switching between text (`grok-2-1212`) and vision (`grok-2-vision-1212`) models based on input type
- **Message Processing**: Conversation history management and context preservation across interactions

## Core Features
- **Multi-modal Input**: Automatic detection and processing of image URLs within text messages
- **Streaming Responses**: Real-time response generation using server-sent events
- **Conversation Memory**: Maintains chat history for contextual conversations
- **Vision Capabilities**: Specialized handling for image analysis with high-detail processing

## Configuration Management
- **Environment Variables**: API keys stored securely using environment variable patterns
- **Model Configuration**: Centralized model selection with fallback options
- **Error Handling**: Graceful handling of API failures and invalid inputs

# External Dependencies

## Core Dependencies
- **gradio**: Web interface framework for creating the chat UI
- **requests**: HTTP client for API communication
- **openai**: Client library for xAI API integration (using OpenAI-compatible endpoints)
- **opencv-python**: Image processing capabilities (opencv-python-headless for server environments)
- **ffmpeg-python**: Multimedia processing support

## External APIs
- **xAI Grok API**: Primary AI service accessible at `https://api.x.ai/v1`
  - Text model: `grok-2-1212`
  - Vision model: `grok-2-vision-1212`
  - Requires API key authentication

## System Dependencies
- **FFmpeg**: Required for multimedia processing (installable via apt in Replit environment)
- **Python 3.x**: Runtime environment with support for async operations and HTTP streaming

## Development Environment
- **Replit**: Designed to run in Replit's cloud environment with built-in dependency management
- **Environment Configuration**: Uses Replit's Secrets feature for secure API key storage