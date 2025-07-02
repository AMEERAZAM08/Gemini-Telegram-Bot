import telebot
from telebot import apihelper
import google.generativeai as genai
import os
import json
import time
from typing import Dict, List
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import requests
import io
import base64
from threading import Thread, Timer, Lock, RLock
from datetime import datetime, timedelta
import hashlib
import tempfile
import traceback
import uuid
import mimetypes
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
from collections import defaultdict, deque

# Import the new genai client separately
try:
    from google import genai as genai_client
    from google.genai import types
    ADVANCED_GENAI_AVAILABLE = True
except ImportError:
    ADVANCED_GENAI_AVAILABLE = False
    print("⚠️  Advanced genai client not available. Using fallback generation.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = os.getenv('BOT_TOKEN', 'YOUR_BOT_TOKEN_HERE')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY_HERE')

# Multi-user configuration
MAX_CONCURRENT_USERS = int(os.getenv('MAX_CONCURRENT_USERS', '100'))
MAX_CONVERSATIONS_PER_USER = int(os.getenv('MAX_CONVERSATIONS_PER_USER', '20'))
USER_SESSION_TIMEOUT_MINUTES = int(os.getenv('USER_SESSION_TIMEOUT_MINUTES', '60'))
MAX_MEMORY_USAGE_MB = int(os.getenv('MAX_MEMORY_USAGE_MB', '500'))

# Enable middleware for telebot
apihelper.ENABLE_MIDDLEWARE = True

# Initialize bot and Gemini
bot = telebot.TeleBot(BOT_TOKEN, threaded=True, num_threads=8)
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Flask app for REST API
app = Flask(__name__)
CORS(app)

# Thread pool for concurrent operations
executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix="BotWorker")

def save_binary_file(file_path: str, data: bytes):
    """Save binary data to a file with file locking"""
    import fcntl
    try:
        with open(file_path, 'wb') as f:
            # Lock file for exclusive access
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(data)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:
        # fcntl not available on Windows, fallback to normal write
        with open(file_path, 'wb') as f:
            f.write(data)

class RateLimiter:
    """Per-user rate limiting"""
    def __init__(self):
        self.user_requests = defaultdict(lambda: deque())
        self.lock = Lock()
        
    def is_allowed(self, user_id: int, max_requests: int = 10, window_seconds: int = 60) -> tuple[bool, str]:
        """Check if user is within rate limits"""
        with self.lock:
            now = time.time()
            user_queue = self.user_requests[user_id]
            
            # Remove old requests outside window
            while user_queue and user_queue[0] < now - window_seconds:
                user_queue.popleft()
            
            # Check if user can make request
            if len(user_queue) >= max_requests:
                return False, f"Rate limit exceeded. Max {max_requests} requests per {window_seconds} seconds."
            
            # Add current request
            user_queue.append(now)
            return True, "Allowed"

class UserSessionManager:
    """Manage user sessions and cleanup"""
    def __init__(self):
        self.user_sessions = {}  # user_id -> last_activity_time
        self.lock = Lock()
        
    def update_activity(self, user_id: int):
        """Update user's last activity time"""
        with self.lock:
            self.user_sessions[user_id] = datetime.now()
    
    def cleanup_inactive_users(self, conversations_dict: dict, contexts_dict: dict, api_keys_dict: dict):
        """Clean up inactive user data"""
        cutoff_time = datetime.now() - timedelta(minutes=USER_SESSION_TIMEOUT_MINUTES)
        inactive_users = []
        
        with self.lock:
            for user_id, last_activity in list(self.user_sessions.items()):
                if last_activity < cutoff_time:
                    inactive_users.append(user_id)
                    del self.user_sessions[user_id]
        
        # Clean up inactive user data
        for user_id in inactive_users:
            conversations_dict.pop(user_id, None)
            contexts_dict.pop(user_id, None)
            api_keys_dict.pop(user_id, None)
            logger.info(f"Cleaned up inactive user {user_id}")
        
        if inactive_users:
            gc.collect()  # Force garbage collection

class AgenticBot:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.vision_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Thread-safe storage for user data
        self.conversations: Dict[int, List[Dict]] = {}
        self.user_contexts: Dict[int, Dict] = {}
        self.user_api_keys: Dict[int, str] = {}
        
        # Thread-safe locks for data structures
        self.conversations_lock = RLock()
        self.contexts_lock = RLock()
        self.api_keys_lock = RLock()
        
        # Rate limiting and session management
        self.rate_limiter = RateLimiter()
        self.session_manager = UserSessionManager()
        
        # Storage for media groups (multiple images)
        self.media_groups: Dict[str, Dict] = {}
        self.media_group_timers: Dict[str, object] = {}
        self.media_groups_lock = Lock()
        
        # Active users tracking
        self.active_users = set()
        self.active_users_lock = Lock()
        
        # Memory-based storage (no local files to avoid permission issues)
        self.memory_storage = {}  # user_id -> list of analysis/generation records
        self.user_data = {}       # user_id -> user statistics and metadata
        self.data_lock = Lock()   # For memory storage thread safety
        
        # Start cleanup thread
        self.cleanup_thread = Thread(target=self._periodic_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        # System prompt for agentic behavior
        self.system_prompt = """You are an intelligent AI assistant integrated into a Telegram bot. Your role is to:

1. Provide helpful, accurate, and contextual responses
2. Remember conversation history within each chat session
3. Ask clarifying questions when needed
4. Offer suggestions and recommendations
5. Maintain a friendly and professional tone
6. Be proactive in helping users achieve their goals

Key behaviors:
- Always consider the conversation context
- If a question is unclear, ask for clarification
- Provide detailed explanations when helpful
- Suggest related topics or follow-up questions
- Remember user preferences mentioned in the conversation
- Be concise but thorough in responses"""

    def _periodic_cleanup(self):
        """Periodic cleanup of inactive users and memory"""
        while True:
            try:
                time.sleep(300)  # Clean up every 5 minutes
                self.session_manager.cleanup_inactive_users(
                    self.conversations, self.user_contexts, self.user_api_keys
                )
                
                # Clean up old media groups
                with self.media_groups_lock:
                    current_time = time.time()
                    expired_groups = [
                        group_id for group_id, group_data in self.media_groups.items()
                        if current_time - group_data.get("timestamp", 0) > 300  # 5 minutes
                    ]
                    
                    for group_id in expired_groups:
                        self.media_groups.pop(group_id, None)
                        if group_id in self.media_group_timers:
                            try:
                                self.media_group_timers[group_id].cancel()
                            except:
                                pass
                            del self.media_group_timers[group_id]
                
                # Memory management and cleanup of old memory storage
                with self.data_lock:
                    for user_id in list(self.memory_storage.keys()):
                        # Keep only last 50 records per user to manage memory
                        if len(self.memory_storage[user_id]) > 50:
                            self.memory_storage[user_id] = self.memory_storage[user_id][-50:]
                
                # Memory monitoring (optional - works without psutil)
                try:
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    if memory_mb > MAX_MEMORY_USAGE_MB:
                        logger.warning(f"High memory usage: {memory_mb:.1f}MB, forcing cleanup")
                        gc.collect()
                except ImportError:
                    # Fallback: force periodic cleanup without memory monitoring
                    logger.info("psutil not available, using periodic cleanup")
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    def check_rate_limit(self, user_id: int) -> tuple[bool, str]:
        """Check if user is within rate limits"""
        return self.rate_limiter.is_allowed(user_id)
    
    def update_user_activity(self, user_id: int):
        """Update user activity and manage active users"""
        self.session_manager.update_activity(user_id)
        
        with self.active_users_lock:
            self.active_users.add(user_id)
            
            # Limit concurrent users
            if len(self.active_users) > MAX_CONCURRENT_USERS:
                # Remove oldest inactive users
                oldest_users = sorted(
                    self.active_users,
                    key=lambda uid: self.session_manager.user_sessions.get(uid, datetime.min)
                )[:len(self.active_users) - MAX_CONCURRENT_USERS]
                
                for old_user in oldest_users:
                    self.active_users.discard(old_user)

    def set_user_api_key(self, user_id: int, api_key: str) -> bool:
        """Set user's custom Gemini API key with thread safety"""
        try:
            # Validate the API key by making a simple test call
            test_genai = genai
            test_genai.configure(api_key=api_key)
            test_model = test_genai.GenerativeModel('gemini-2.0-flash-exp')
            
            # Test with a simple prompt
            test_response = test_model.generate_content("Hello")
            
            if test_response:
                with self.api_keys_lock:
                    self.user_api_keys[user_id] = api_key
                self.update_user_context(user_id, "custom_api_key", "set")
                logger.info(f"User {user_id} successfully set custom API key")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error validating API key for user {user_id}: {e}")
            return False
    
    def get_user_api_key(self, user_id: int) -> str:
        """Get user's API key or default to global key"""
        with self.api_keys_lock:
            return self.user_api_keys.get(user_id, GEMINI_API_KEY)
    
    def has_custom_api_key(self, user_id: int) -> bool:
        """Check if user has set a custom API key"""
        with self.api_keys_lock:
            return user_id in self.user_api_keys
    
    def remove_user_api_key(self, user_id: int):
        """Remove user's custom API key"""
        with self.api_keys_lock:
            if user_id in self.user_api_keys:
                del self.user_api_keys[user_id]
        self.update_user_context(user_id, "custom_api_key", "removed")
        logger.info(f"Removed custom API key for user {user_id}")
    
    def get_user_model(self, user_id: int, model_name: str = 'gemini-2.0-flash-exp'):
        """Get a Gemini model configured with user's API key"""
        user_api_key = self.get_user_api_key(user_id)
        
        # Create model with user's configuration
        temp_genai = genai
        temp_genai.configure(api_key=user_api_key)
        model = temp_genai.GenerativeModel(model_name)
        
        return model, user_api_key
    
    def get_conversation_history(self, user_id: int) -> List[Dict]:
        """Get conversation history for a user"""
        with self.conversations_lock:
            return self.conversations.get(user_id, []).copy()
    
    def add_to_conversation(self, user_id: int, role: str, content: str):
        """Add a message to conversation history with thread safety"""
        with self.conversations_lock:
            if user_id not in self.conversations:
                self.conversations[user_id] = []
            
            self.conversations[user_id].append({
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
            
            # Keep only last messages to manage memory
            max_messages = MAX_CONVERSATIONS_PER_USER
            if len(self.conversations[user_id]) > max_messages:
                self.conversations[user_id] = self.conversations[user_id][-max_messages:]
    
    def get_user_context(self, user_id: int) -> Dict:
        """Get user context and preferences"""
        with self.contexts_lock:
            return self.user_contexts.get(user_id, {}).copy()
    
    def update_user_context(self, user_id: int, key: str, value: str):
        """Update user context with thread safety"""
        with self.contexts_lock:
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = {}
            self.user_contexts[user_id][key] = value
    
    def build_context_prompt(self, user_id: int, current_message: str) -> str:
        """Build a contextual prompt with conversation history"""
        history = self.get_conversation_history(user_id)
        context = self.get_user_context(user_id)
        
        prompt_parts = [self.system_prompt]
        
        if context:
            prompt_parts.append(f"\nUser Context: {json.dumps(context, indent=2)}")
        
        if history:
            prompt_parts.append("\nConversation History:")
            for msg in history[-10:]:  # Last 10 messages
                role_label = "User" if msg["role"] == "user" else "Assistant"
                prompt_parts.append(f"{role_label}: {msg['content']}")
        
        prompt_parts.append(f"\nCurrent User Message: {current_message}")
        prompt_parts.append("\nProvide a helpful, contextual response:")
        
        return "\n".join(prompt_parts)
    
    async def generate_response(self, user_id: int, message: str) -> str:
        """Generate an intelligent response using Gemini"""
        try:
            # Build contextual prompt
            full_prompt = self.build_context_prompt(user_id, message)
            
            # Get user's model with their API key
            user_model, _ = self.get_user_model(user_id)
            
            # Generate response
            response = user_model.generate_content(full_prompt)
            
            if response.text:
                # Add to conversation history
                self.add_to_conversation(user_id, "user", message)
                self.add_to_conversation(user_id, "assistant", response.text)
                
                # Format response for better Telegram UI
                formatted_response = self.format_text_response(response.text, "conversation", user_id)
                return formatted_response
            else:
                return "I'm sorry, I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def download_image_from_url(self, url: str) -> Image.Image:
        """Download and return PIL Image from URL"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            raise
    
    def process_base64_image(self, base64_string: str) -> Image.Image:
        """Process base64 encoded image and return PIL Image"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            return Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error(f"Error processing base64 image: {e}")
            raise
    

    

    

    
    def generate_advanced_gemini(self, text: str, file_name: str = None, model: str = "gemini-2.0-flash-exp", user_id: int = None) -> tuple[str, str]:
        """Generate text or image using advanced Gemini API"""
        if not ADVANCED_GENAI_AVAILABLE:
            # Fallback to regular text generation
            try:
                if user_id:
                    user_model, _ = self.get_user_model(user_id)
                    response = user_model.generate_content(text)
                else:
                    response = self.model.generate_content(text)
                return None, response.text if response.text else "I can help you with text responses."
            except Exception as e:
                logger.error(f"Error in fallback generation: {e}")
                return None, "I encountered an error processing your request."
        
        try:
            # Get the appropriate API key
            api_key = self.get_user_api_key(user_id) if user_id else GEMINI_API_KEY
            
            # Initialize client using the API key
            client = genai_client.Client(api_key=api_key)
            
            # Prepare contents
            content_parts = []
            
            # Add file if provided
            if file_name and os.path.exists(file_name):
                try:
                    # Method 1: Use correct file parameter for new API
                    uploaded_file = client.files.upload(file=file_name)
                    content_parts.append(
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type,
                        )
                    )
                    logger.info(f"Successfully uploaded file: {file_name}")
                except Exception as upload_error:
                    logger.error(f"Error uploading file {file_name}: {upload_error}")
                    logger.error(f"Upload error traceback: {traceback.format_exc()}")
                    try:
                        # Method 2: Try using file bytes directly
                        import mimetypes
                        mime_type = mimetypes.guess_type(file_name)[0] or 'image/png'
                        
                        with open(file_name, 'rb') as f:
                            file_data = f.read()
                        
                        # Create a temporary file-like object
                        import io
                        file_obj = io.BytesIO(file_data)
                        file_obj.name = os.path.basename(file_name)
                        
                        uploaded_file = client.files.upload(file=file_obj)
                        content_parts.append(
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type,
                            )
                        )
                        logger.info(f"Successfully uploaded file using method 2: {file_name}")
                    except Exception as fallback_error:
                        logger.error(f"Method 2 upload failed for {file_name}: {fallback_error}")
                        logger.error(f"Method 2 error traceback: {traceback.format_exc()}")
                        try:
                            # Method 3: Use legacy genai.upload_file if available
                            import google.generativeai as genai_legacy
                            genai_legacy.configure(api_key=api_key)
                            uploaded_file = genai_legacy.upload_file(file_name)
                            content_parts.append(
                                types.Part.from_uri(
                                    file_uri=uploaded_file.uri,
                                    mime_type=uploaded_file.mime_type,
                                )
                            )
                            logger.info(f"Successfully uploaded file using legacy method: {file_name}")
                        except Exception as legacy_error:
                            logger.error(f"Legacy upload also failed for {file_name}: {legacy_error}")
                            logger.error(f"Legacy error traceback: {traceback.format_exc()}")
                            # Skip file upload and continue with text only
                            logger.warning(f"Skipping file upload for {file_name}, continuing with text prompt only")
                            pass
            
            # Add text prompt
            content_parts.append(types.Part.from_text(text=text))
            
            contents = [
                types.Content(
                    role="user",
                    parts=content_parts,
                ),
            ]
            
            # Configure for both text and image generation
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_modalities=["image", "text"],
                response_mime_type="text/plain",
            )

            text_response = ""
            image_path = None
            
            # Create a temporary file to potentially store image data
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_path = tmp.name
                
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                        continue
                    
                    candidate = chunk.candidates[0].content.parts[0]
                    
                    # Check for inline image data
                    if candidate.inline_data:
                        save_binary_file(temp_path, candidate.inline_data.data)
                        logger.info(f"File of mime type {candidate.inline_data.mime_type} saved to: {temp_path} and prompt input: {text}")
                        image_path = temp_path
                        # If an image is found, we assume that is the desired output
                        break
                    else:
                        # Accumulate text response if no inline_data is present
                        if hasattr(chunk, 'text') and chunk.text:
                            text_response += chunk.text
            
            # Clean up uploaded files - handled automatically by the API
            # No manual cleanup needed for uploaded files
            
            return image_path, text_response
            
        except Exception as e:
            logger.error(f"Error in advanced Gemini generation: {e}")
            # Fallback to regular generation
            try:
                if user_id:
                    user_model, _ = self.get_user_model(user_id)
                    response = user_model.generate_content(text)
                else:
                    response = self.model.generate_content(text)
                return None, response.text if response.text else "I encountered an error with advanced generation, but I can still help with text responses."
            except:
                return None, f"I encountered an error while processing your request: {str(e)}"

    async def generate_image_with_gemini(self, user_id: int, prompt: str, reference_image: Image.Image = None) -> tuple[str, str]:
        """Generate image using Gemini with optional reference image"""
        try:
            temp_file_path = None
            
            # Save reference image to temporary file if provided
            if reference_image:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    temp_file_path = tmp.name
                    reference_image.save(temp_file_path)
            
            # Use advanced generation
            image_path, text_response = self.generate_advanced_gemini(
                text=prompt,
                file_name=temp_file_path,
                user_id=user_id
            )
            
            # Clean up temporary reference file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            
            return image_path, text_response
                
        except Exception as e:
            logger.error(f"Error generating with Gemini: {e}")
            return None, f"I encountered an error while processing your request: {str(e)}"
    
    def format_text_response(self, text: str, response_type: str = "general", user_id: int = None) -> str:
        """Format text response for better Telegram UI presentation"""
        if not text or not text.strip():
            return text
        
        # Check user preference for formatting
        if user_id:
            user_context = self.get_user_context(user_id)
            text_format = user_context.get("text_format", "enhanced")  # Default to enhanced
            if text_format == "plain":
                return text.strip()  # Return plain text if user prefers it
        
        # Clean up the text
        formatted_text = text.strip()
        
        # Handle different response types
        if response_type == "technical":
            # For technical/code responses
            if any(keyword in text.lower() for keyword in ["code", "function", "class", "import", "def ", "var ", "const "]):
                # Wrap code-like content in code blocks
                formatted_text = f"```\n{formatted_text}\n```"
        
        # Format lists and numbered items
        lines = formatted_text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
            
            # Convert basic list items to better formatting
            if line.startswith('- ') or line.startswith('* '):
                # Convert to bullet points with emojis
                content = line[2:].strip()
                formatted_lines.append(f"• {content}")
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                # Keep numbered lists but with better spacing
                formatted_lines.append(f"**{line}**")
            elif line.endswith(':') and len(line) < 50:
                # Section headers
                formatted_lines.append(f"**{line}**\n")
            else:
                # Regular text - check for emphasis words
                emphasized_line = line
                
                # Add emphasis to important words
                emphasis_words = {
                    r'\b(important|note|warning|tip|key|main|primary|essential)\b': r'**\1**',
                    r'\b(however|but|although|nevertheless|therefore|thus|hence)\b': r'*\1*',
                }
                
                for pattern, replacement in emphasis_words.items():
                    emphasized_line = re.sub(pattern, replacement, emphasized_line, flags=re.IGNORECASE)
                
                formatted_lines.append(emphasized_line)
        
        # Join lines with proper spacing
        formatted_text = '\n'.join(formatted_lines)
        
        # Add proper spacing around sections
        formatted_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', formatted_text)
        
        # Add visual separators for long responses
        if len(formatted_text) > 500 and response_type == "explanation":
            # Add a visual break for long explanations
            formatted_text = f"💭 **Response:**\n\n{formatted_text}\n\n✨ *Hope this helps!*"
        elif response_type == "analysis":
            formatted_text = f"🔍 **Analysis:**\n\n{formatted_text}"
        elif response_type == "conversation":
            # Just add a subtle indicator for conversation responses
            if len(formatted_text) > 200:
                formatted_text = f"💬 {formatted_text}"
        
        return formatted_text



    def add_to_media_group(self, media_group_id: str, user_id: int, message, image: Image.Image, caption: str = None):
        """Add an image to a media group for batch processing with thread safety"""
        with self.media_groups_lock:
            if media_group_id not in self.media_groups:
                self.media_groups[media_group_id] = {
                    "user_id": user_id,
                    "chat_id": message.chat.id,
                    "images": [],
                    "captions": [],
                    "messages": [],
                    "timestamp": time.time()
                }
            
            self.media_groups[media_group_id]["images"].append(image)
            self.media_groups[media_group_id]["captions"].append(caption or "")
            self.media_groups[media_group_id]["messages"].append(message)
        
        logger.info(f"Added image to media group {media_group_id}. Total images: {len(self.media_groups[media_group_id]['images'])}")

    def process_media_group(self, media_group_id: str):
        """Process all images in a media group"""
        if media_group_id not in self.media_groups:
            logger.warning(f"Media group {media_group_id} not found")
            return
        
        group_data = self.media_groups[media_group_id]
        user_id = group_data["user_id"]
        chat_id = group_data["chat_id"]
        images = group_data["images"]
        captions = group_data["captions"]
        messages = group_data["messages"]
        
        try:
            logger.info(f"Processing media group {media_group_id} with {len(images)} images")
            
            # Show typing indicator
            bot.send_chat_action(chat_id, 'upload_photo')
            
            # Get the main caption (from first image with caption or first image)
            main_caption = next((cap for cap in captions if cap), "") or f"Transform these {len(images)} images"
            
            # Generate from all images
            self.process_multiple_generation(user_id, chat_id, images, captions, messages, main_caption)
        
        except Exception as e:
            logger.error(f"Error processing media group {media_group_id}: {e}")
            bot.send_message(chat_id, f"❌ Error processing images: {str(e)}")
        
        finally:
            # Clean up media group data
            if media_group_id in self.media_groups:
                del self.media_groups[media_group_id]
            if media_group_id in self.media_group_timers:
                del self.media_group_timers[media_group_id]



    def process_multiple_generation(self, user_id: int, chat_id: int, images: List[Image.Image], captions: List[str], messages: List, main_caption: str):
        """Generate from multiple images using combined Gemini processing"""
        try:
            if len(images) > 1:
                bot.send_message(chat_id, f"🎨 Processing {len(images)} images together...")
            else:
                bot.send_message(chat_id, f"🎨 Processing your image...")
            bot.send_chat_action(chat_id, 'upload_photo')
            
            # Save all images temporarily for Gemini upload
            temp_file_paths = []
            
            for i, image in enumerate(images):
                # Save temporarily for Gemini upload
                temp_file = tempfile.NamedTemporaryFile(suffix=f"_{i}.png", delete=False)
                temp_file_path = temp_file.name
                temp_file.close()
                image.save(temp_file_path)
                temp_file_paths.append(temp_file_path)
            
            try:
                # Create combined prompt for all images
                if main_caption and main_caption.strip():
                    combined_prompt = f"Process these {len(images)} images together: {main_caption}"
                else:
                    combined_prompt = f"Transform these {len(images)} images into enhanced, more realistic versions. Consider them as a group and create cohesive results."
                
                # Add individual captions if they exist
                individual_prompts = []
                for i, caption in enumerate(captions):
                    if caption and caption.strip():
                        individual_prompts.append(f"Image {i+1}: {caption}")
                
                if individual_prompts:
                    combined_prompt += f"\n\nIndividual instructions:\n" + "\n".join(individual_prompts)
                
                logger.info(f"Processing {len(images)} images together with prompt: {combined_prompt[:200]}...")
                
                # Use the combined generation approach
                result_images, text_responses = self.generate_from_multiple_images(
                    temp_file_paths, combined_prompt, user_id
                )
                
                if result_images:
                    # Send generated images
                    for i, image_path in enumerate(result_images):
                        try:
                            with open(image_path, 'rb') as img_file:
                                bot.send_photo(
                                    chat_id,
                                    img_file,
                                    caption=f"🎨 Generated result {i+1}/{len(result_images)}" + (f"\n📸 From {len(images)} images" if len(images) > 1 else "")
                                )
                            
                            # Clean up generated file
                            os.unlink(image_path)
                            
                        except Exception as e:
                            logger.error(f"Error sending generated image {i+1}: {e}")
                
                elif text_responses:
                    # If we got text responses instead of images
                    formatted_response = self.format_text_response(text_responses, "explanation", user_id)
                    response_header = "🤖 **Generated Response:**" if len(images) == 1 else f"🤖 **Generated Response from {len(images)} images:**"
                    bot.send_message(
                        chat_id,
                        f"{response_header}\n\n{formatted_response}",
                        parse_mode='Markdown'
                    )
                
                # Send completion message if multiple results
                if len(result_images) > 1:
                    completion_msg = f"✅ **Processing Complete!**\n\n📸 Generated {len(result_images)} images from your {len(images)} photos"
                    formatted_msg = self.format_text_response(completion_msg, "explanation", user_id)
                    bot.send_message(chat_id, formatted_msg, parse_mode='Markdown')
                
            finally:
                # Clean up all temporary files
                for temp_path in temp_file_paths:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            bot.send_message(chat_id, f"❌ Error generating from images: {str(e)}")

    def generate_from_multiple_images(self, file_paths: List[str], prompt: str, user_id: int = None) -> tuple[List[str], str]:
        """Generate using multiple images uploaded to Gemini together"""
        if not ADVANCED_GENAI_AVAILABLE:
            logger.warning("Advanced GenAI not available, falling back to individual processing")
            return [], "Advanced image generation not available. Please try with regular processing mode."
        
        try:
            # Get the appropriate API key
            api_key = self.get_user_api_key(user_id) if user_id else GEMINI_API_KEY
            
            # Initialize client
            client = genai_client.Client(api_key=api_key)
            
            # Upload all files to Gemini
            uploaded_files = []
            content_parts = []
            
            for i, file_path in enumerate(file_paths):
                try:
                    logger.info(f"Uploading file {i+1}/{len(file_paths)}: {file_path}")
                    # Method 1: Use correct file parameter for new API
                    uploaded_file = client.files.upload(file=file_path)
                    uploaded_files.append(uploaded_file)
                    
                    content_parts.append(
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type,
                        )
                    )
                    logger.info(f"Successfully uploaded file {i+1}: {uploaded_file.uri}")
                    
                except Exception as upload_error:
                    logger.error(f"Method 1 upload failed for {file_path}: {upload_error}")
                    logger.error(f"Upload error traceback: {traceback.format_exc()}")
                    try:
                        # Method 2: Try using file bytes directly
                        import mimetypes
                        mime_type = mimetypes.guess_type(file_path)[0] or 'image/png'
                        
                        with open(file_path, 'rb') as f:
                            file_data = f.read()
                        
                        # Create a temporary file-like object
                        import io
                        file_obj = io.BytesIO(file_data)
                        file_obj.name = os.path.basename(file_path)
                        
                        uploaded_file = client.files.upload(file=file_obj)
                        uploaded_files.append(uploaded_file)
                        
                        content_parts.append(
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type,
                            )
                        )
                        logger.info(f"Successfully uploaded file {i+1} using method 2: {uploaded_file.uri}")
                        
                    except Exception as fallback_error:
                        logger.error(f"Method 2 upload failed for {file_path}: {fallback_error}")
                        logger.error(f"Method 2 error traceback: {traceback.format_exc()}")
                        try:
                            # Method 3: Use legacy genai.upload_file if available
                            import google.generativeai as genai_legacy
                            genai_legacy.configure(api_key=api_key)
                            uploaded_file = genai_legacy.upload_file(file_path)
                            uploaded_files.append(uploaded_file)
                            
                            content_parts.append(
                                types.Part.from_uri(
                                    file_uri=uploaded_file.uri,
                                    mime_type=uploaded_file.mime_type,
                                )
                            )
                            logger.info(f"Successfully uploaded file {i+1} using legacy method: {uploaded_file.uri}")
                            
                        except Exception as legacy_error:
                            logger.error(f"All upload methods failed for {file_path}: {legacy_error}")
                            logger.error(f"Legacy error traceback: {traceback.format_exc()}")
                            # Skip this file and continue with others
                            logger.warning(f"Skipping file {file_path}, continuing with remaining files")
                            continue
            
            if not content_parts:
                return [], "Failed to upload any images to Gemini."
            
            # Add the text prompt
            content_parts.append(types.Part.from_text(text=prompt))
            
            contents = [
                types.Content(
                    role="user",
                    parts=content_parts,
                ),
            ]
            
            # Configure for both text and image generation
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_modalities=["image", "text"],
                response_mime_type="text/plain",
            )
            
            generated_images = []
            text_response = ""
            
            logger.info(f"Starting generation with {len(uploaded_files)} uploaded files")
            
            # Process the response stream
            for chunk in client.models.generate_content_stream(
                model="gemini-2.0-flash-exp",
                contents=contents,
                config=generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                
                candidate = chunk.candidates[0].content.parts[0]
                
                # Check for inline image data
                if candidate.inline_data:
                    # Save generated image
                    temp_gen_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    temp_gen_path = temp_gen_file.name
                    temp_gen_file.close()
                    
                    save_binary_file(temp_gen_path, candidate.inline_data.data)
                    generated_images.append(temp_gen_path)
                    logger.info(f"Generated image saved to: {temp_gen_path}")
                else:
                    # Accumulate text response
                    if hasattr(chunk, 'text') and chunk.text:
                        text_response += chunk.text
            
            logger.info(f"Generation complete. Images: {len(generated_images)}, Text length: {len(text_response)}")
            return generated_images, text_response
            
        except Exception as e:
            logger.error(f"Error in generate_from_multiple_images: {e}")
            return [], f"Error processing images: {str(e)}"

    def get_user_data(self, user_id: int) -> Dict:
        """Get user data from memory storage"""
        with self.data_lock:
            if user_id not in self.user_data:
                self.user_data[user_id] = {
                    "images_processed": 0,
                    "analysis_count": 0,
                    "generated_count": 0,
                    "session_start": datetime.now().isoformat(),
                    "last_activity": datetime.now().isoformat()
                }
            return self.user_data[user_id]
    
    def update_user_stats(self, user_id: int, operation: str):
        """Update user statistics in memory"""
        with self.data_lock:
            user_data = self.get_user_data(user_id)
            if operation == "image_processed":
                user_data["images_processed"] += 1
            elif operation == "analysis":
                user_data["analysis_count"] += 1
            elif operation == "generation":
                user_data["generated_count"] += 1
            user_data["last_activity"] = datetime.now().isoformat()
    
    def save_analysis_in_memory(self, user_id: int, prompt: str, response: str, operation_type: str = "analysis") -> str:
        """Save analysis response in memory and return a unique identifier"""
        try:
            with self.data_lock:
                # Create unique identifier
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:6]
                record_id = f"{operation_type}_{timestamp}_{unique_id}"
                
                # Store in memory
                if user_id not in self.memory_storage:
                    self.memory_storage[user_id] = []
                
                record = {
                    "id": record_id,
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "prompt": prompt,
                    "response": response,
                    "operation_type": operation_type,
                    "metadata": {
                        "response_length": len(response),
                        "word_count": len(response.split()),
                        "processing_time": time.time()
                    }
                }
                
                self.memory_storage[user_id].append(record)
                
                # Keep only last 100 records per user to manage memory
                if len(self.memory_storage[user_id]) > 100:
                    self.memory_storage[user_id] = self.memory_storage[user_id][-100:]
                
                # Update user stats
                self.update_user_stats(user_id, operation_type)
                
                logger.info(f"Analysis saved in memory: {record_id}")
                return record_id
        except Exception as e:
            logger.error(f"Error saving analysis for user {user_id}: {e}")
            return ""
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get statistics for a user's data"""
        try:
            user_data = self.get_user_data(user_id)
            
            # Count records in memory
            memory_records = len(self.memory_storage.get(user_id, []))
            
            return {
                "user_id": user_id,
                "images_processed": user_data.get("images_processed", 0),
                "analysis_count": user_data.get("analysis_count", 0),
                "generated_count": user_data.get("generated_count", 0),
                "memory_records": memory_records,
                "session_start": user_data.get("session_start"),
                "last_activity": user_data.get("last_activity"),
                "storage_type": "memory"
            }
        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {"error": str(e)}

# Initialize the agentic bot
agentic_bot = AgenticBot()

# Decorator for rate limiting and user activity tracking
def with_rate_limit(func):
    """Decorator to add rate limiting and user activity tracking to handlers"""
    def wrapper(message):
        user_id = message.from_user.id
        
        # Check rate limit
        allowed, reason = agentic_bot.check_rate_limit(user_id)
        if not allowed:
            bot.reply_to(message, f"⚠️ {reason}")
            return
        
        # Update user activity
        agentic_bot.update_user_activity(user_id)
        
        # Determine timeout based on operation type
        timeout = 5000  # Default timeout
        
        # Increase timeout for image-related operations
        if hasattr(message, 'content_type'):
            if message.content_type == 'photo':
                timeout = 5000  # 2 minutes for photo processing
            elif message.content_type in ['document', 'audio', 'video', 'voice']:
                timeout = 5000  # 3 minutes for media processing
        
        # Check for image generation commands
        if hasattr(message, 'text') and message.text:
            image_keywords = ['/generate', '/demo', 'draw', 'create', 'make', 'generate']
            if any(keyword in message.text.lower() for keyword in image_keywords):
                timeout = 5000  # 2 minutes for generation commands
        
        # Execute in thread pool for better concurrency
        try:
            return executor.submit(func, message).result(timeout=timeout)
        except Exception as e:
            import traceback
            logger.error(f"Error in handler {func.__name__} for user {user_id}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Better error messages based on exception type
            if "TimeoutError" in str(type(e)):
                error_msg = f"⏰ Operation timed out after {timeout} seconds. The request may be taking longer than expected. Please try again with a simpler request."
            else:
                error_msg = f"❌ I encountered an error processing your request. Please try again.\n\nError: {str(e)[:100]}..."
            
            bot.reply_to(message, error_msg)
    
    return wrapper

# Multi-user status tracking
def get_bot_status():
    """Get current bot status for multiple users"""
    with agentic_bot.active_users_lock:
        active_count = len(agentic_bot.active_users)
    
    with agentic_bot.conversations_lock:
        total_conversations = len(agentic_bot.conversations)
    
    return {
        "active_users": active_count,
        "total_users": total_conversations,
        "max_concurrent": MAX_CONCURRENT_USERS
    }

# Bot handlers
@bot.message_handler(commands=['start'])
@with_rate_limit
def handle_start(message):
    user_id = message.from_user.id
    username = message.from_user.first_name or "there"
    
    welcome_message = f"""👋 Hello {username}! I'm your intelligent AI assistant powered by Google Gemini.

I can help you with:
🤔 Answering questions on any topic
💡 Providing explanations and insights
🎨 Automatically generating images from text
📚 Research and learning support
🛠️ Problem-solving assistance
💬 Engaging conversations

✨ **New Features**:
• Type anything → I'll automatically respond with text or images!
• Send photos → I'll generate enhanced/transformed versions!
• Send multiple photos → I'll process all of them automatically!

Examples:
• "Draw a sunset" → I'll create an image
• "Explain quantum physics" → Text explanation
• Send anime photo → I'll make it realistic
• Send 5 photos → Gemini will process all 5 together for cohesive results


I remember our conversation context, so feel free to ask follow-up questions!

How can I assist you today?"""
    
    bot.reply_to(message, welcome_message)
    
    # Initialize user context
    agentic_bot.update_user_context(user_id, "name", username)
    agentic_bot.update_user_context(user_id, "started_at", time.strftime("%Y-%m-%d %H:%M:%S"))

@bot.message_handler(commands=['help'])
@with_rate_limit
def handle_help(message):
    help_text = """🤖 **AI Assistant Help**

**📱 Basic Commands:**
• /start - Start conversation and get welcome message
• /help - Show this help message (current command)
• /clear - Clear conversation history
• /context - Show your current context and preferences
• /setpref - Set user preferences
• /stats - Show your storage statistics

**🎨 Image & Generation Commands:**
• /demo - Demo image generation
• /generate - Generate an image from text description
• /format - Test text formatting features
• /multitest - Test multiple image functionality

**📋 Information Commands:**
• /rules - View bot rules and guidelines
• /about - Bot information and features
• /status - Your usage statistics and preferences
• /reset - Reset all your data and preferences
• /ping - Check bot responsiveness
• /version - Bot version and system info
• /feedback - Send feedback to developers
• /limit - View current rate limits
• /privacy - Privacy policy and data handling
• /export - Export your conversation history
• /config - Show all commands and configuration

**🔑 API Key Commands:**
• /setkey - Set your own Gemini API key for personal quota
• /keystatus - Check your current API key status
• /removekey - Remove your custom key and use shared bot key

**🤖 System Commands:**
• /botstatus - View multi-user bot status and performance

**✨ Key Features:**
✅ Contextual conversations with memory
✅ Intelligent responses powered by Gemini AI
✅ Automatic image generation from text
✅ Smart image generation from text prompts
✅ Smart auto text/image responses
✅ Enhanced text formatting and typography
✅ Multi-topic support and follow-up questions
✅ Personalized interactions with preferences

**🎨 Usage:**
💬 Just type anything - I will automatically respond with image or text
📸 Send any photo - I will generate an enhanced version
📷 Send multiple photos - Gemini will process them together
🎨 Use /generate for explicit image creation

Type any message or send an image to get started! 🚀"""
    
    bot.reply_to(message, help_text, parse_mode='Markdown')

@bot.message_handler(commands=['clear'])
@with_rate_limit
def handle_clear(message):
    user_id = message.from_user.id
    with agentic_bot.conversations_lock:
        agentic_bot.conversations[user_id] = []
    bot.reply_to(message, "🧹 Conversation history cleared! We can start fresh.")

@bot.message_handler(commands=['context'])
def handle_context(message):
    user_id = message.from_user.id
    context = agentic_bot.get_user_context(user_id)
    history_count = len(agentic_bot.get_conversation_history(user_id))
    
    if context:
        context_text = "📋 **Your Current Context:**\n\n"
        for key, value in context.items():
            if key == "analysis_style":
                context_text += f"• **Analysis Style:** {value} (change with `/setpref analysis_style brief/detailed`)\n"
            elif key == "text_format":
                context_text += f"• **Text Format:** {value} (change with `/setpref text_format enhanced/plain`)\n"
            else:
                context_text += f"• **{key.title()}:** {value}\n"
        
        # Show default preferences if not set
        if "analysis_style" not in context:
            context_text += "• **Analysis Style:** brief (default) - change with `/setpref analysis_style detailed`\n"
        if "text_format" not in context:
            context_text += "• **Text Format:** enhanced (default) - change with `/setpref text_format plain`\n"
            
        context_text += f"\n💬 **Messages in history:** {history_count}"
        context_text += f"\n\n💡 **Tips:**\n• Use `/setpref analysis_style brief/detailed` for image analysis style\n• Use `/setpref text_format enhanced/plain` for text formatting\n• Try `/format` to test enhanced typography"
    else:
        context_text = "No context information available yet. Start chatting to build context!\n\n💡 **Tip:** Use `/setpref analysis_style brief` for concise image analysis (default) or `detailed` for comprehensive descriptions."
    
    bot.reply_to(message, context_text, parse_mode='Markdown')

@bot.message_handler(commands=['setpref'])
def handle_setpref(message):
    user_id = message.from_user.id
    try:
        # Extract preference from command
        parts = message.text.split(' ', 2)
        if len(parts) >= 3:
            pref_key = parts[1]
            pref_value = parts[2]
            agentic_bot.update_user_context(user_id, pref_key, pref_value)
            
            # Special handling for specific preferences
            if pref_key == "analysis_style":
                if pref_value.lower() in ["brief", "detailed"]:
                    bot.reply_to(message, f"✅ Analysis style set to **{pref_value}**!\n\n📸 Your image analysis will now be {'concise (1-3 sentences)' if pref_value.lower() == 'brief' else 'detailed and comprehensive'}.", parse_mode='Markdown')
                else:
                    bot.reply_to(message, f"✅ Preference updated: {pref_key} = {pref_value}\n\n💡 For analysis style, use 'brief' or 'detailed'", parse_mode='Markdown')
            elif pref_key == "text_format":
                if pref_value.lower() in ["enhanced", "plain"]:
                    bot.reply_to(message, f"✅ Text formatting set to **{pref_value}**!\n\n📝 Your text responses will now be {'enhanced with better typography' if pref_value.lower() == 'enhanced' else 'plain without special formatting'}.", parse_mode='Markdown')
                else:
                    bot.reply_to(message, f"✅ Preference updated: {pref_key} = {pref_value}\n\n💡 For text format, use 'enhanced' or 'plain'", parse_mode='Markdown')
            else:
                bot.reply_to(message, f"✅ Preference updated: {pref_key} = {pref_value}", parse_mode='Markdown')
        else:
            bot.reply_to(message, "Usage: /setpref <key> <value>\n\nExamples:\n• `/setpref analysis_style brief` - Short image analysis\n• `/setpref analysis_style detailed` - Comprehensive analysis\n• `/setpref text_format enhanced` - Better text typography\n• `/setpref text_format plain` - Plain text formatting\n• `/setpref language English` - Set language preference", parse_mode='Markdown')
    except Exception as e:
        bot.reply_to(message, f"Error setting preference: {str(e)}")

@bot.message_handler(commands=['stats'])
@with_rate_limit
def handle_stats(message):
    user_id = message.from_user.id
    try:
        stats = agentic_bot.get_user_stats(user_id)
        
        if "error" in stats:
            bot.reply_to(message, f"Error getting stats: {stats['error']}")
            return
        
        stats_text = f"""📊 **Your Storage Stats:**

        🖼️ **Images Processed:** {stats.get('images_processed', 0)}
        📝 **Analysis Count:** {stats.get('analysis_count', 0)}
        🎨 **Generated Count:** {stats.get('generated_count', 0)}
        💾 **Memory Records:** {stats.get('memory_records', 0)}

💡 All your analysis and generation activities are logged in memory for this session!"""
        
        bot.reply_to(message, stats_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in stats command: {e}")
        bot.reply_to(message, f"Error getting statistics: {str(e)}")

@bot.message_handler(commands=['demo'])
@with_rate_limit
def handle_demo_command(message):
    """Demonstrate image generation"""
    try:
        user_id = message.from_user.id
        
        # Show typing indicator
        bot.send_chat_action(message.chat.id, 'upload_photo')
        
        # Generate a demo image
        demo_prompt = "Create a beautiful landscape with mountains and a sunset"
        
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        image_path, text_response = loop.run_until_complete(
            agentic_bot.generate_image_with_gemini(user_id, demo_prompt)
        )
        
        if image_path:
            # Send the generated image
            with open(image_path, 'rb') as img_file:
                bot.send_photo(
                    message.chat.id, 
                    img_file, 
                    caption=f"🎨 Demo generated image: {demo_prompt}"
                )
            
            # Clean up temporary file
            try:
                os.unlink(image_path)
            except:
                pass
        else:
            # Send text response with formatting
            formatted_response = agentic_bot.format_text_response(text_response, "explanation", user_id)
            bot.reply_to(message, f"🤖 **Demo Response:**\n\n{formatted_response}", parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in demo command: {e}")
        bot.reply_to(message, "Sorry, I couldn't run the demo. Please try again.")



@bot.message_handler(commands=['format'])
def handle_format_test(message):
    """Test text formatting with examples"""
    try:
        test_text = """Here's a sample response to test formatting:

Important: This is a key point that should be emphasized.

Features:
- Automatic bullet point conversion
- Bold headers and emphasis
- Code block detection
- Better line spacing

Note: The formatting improves readability.

However, it maintains the original content while enhancing presentation.

Example list:
1. First numbered item
2. Second numbered item  
3. Third numbered item

Technical example:
def function_example():
    return "code detected"

This demonstrates the enhanced typography for better user experience."""
        
        formatted_response = agentic_bot.format_text_response(test_text, "explanation")
        bot.reply_to(message, f"📝 **Text Formatting Demo:**\n\n{formatted_response}", parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in format test: {e}")
        bot.reply_to(message, "Sorry, I couldn't run the formatting test.")

@bot.message_handler(commands=['multitest'])
def handle_multi_test(message):
    """Test multiple image functionality"""
    try:
        help_text = """📷 **Multiple Image Support Test**

**How to test:**
1. Select multiple photos in your gallery (2-10 images)
2. Send them all at once to this chat
3. The bot will automatically detect it's a media group
4. Processing will begin after 2 seconds

**What happens:**
• **Default mode:** All images uploaded to Gemini together for cohesive generation
• **Analysis mode:** Combined analysis of all images in one response
• **With caption:** Custom processing applied to all images as a group

**Try these:**
• Send 3 photos → Gemini processes all 3 together for related results
• Use `/analyze` then send 3 photos → Get combined analysis of all images
• Send photos with caption "make them cartoonish" → All images transformed together
• Send related photos → Gemini considers them as a set for better results

**New Combined Processing:**
✅ All images uploaded to Gemini simultaneously
✅ Gemini processes images as a group for cohesive results
✅ Smart prompt combination with individual captions
✅ Better results for related/sequential images
✅ All images and results saved to your storage

Ready to test? Send multiple photos now! 🚀"""
        
        formatted_response = agentic_bot.format_text_response(help_text, "explanation")
        bot.reply_to(message, formatted_response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in multitest command: {e}")
        bot.reply_to(message, "Sorry, I couldn't show the multiple image test info.")

@bot.message_handler(commands=['config', 'configure', 'commands'])
def handle_config_command(message):
    """Show all available commands and configuration"""
    try:
        config_text = f"""🤖 **AGENTIC GEMINI BOT CONFIGURATION**

**📱 Available Commands:**
/start - Initialize bot and get welcome message
/help - Show detailed help and all features  
/clear - Clear conversation history
/context - View current user context and preferences
/setpref <key> <value> - Set user preferences
/stats - View local storage statistics
/demo - Demo image generation
/generate <prompt> - Generate images from text
/analyze - Switch next photo to analysis mode
/format - Test text formatting features
/multitest - Test multiple image functionality
/config - Show this configuration (current command)

**🆕 New Commands:**
/rules - View bot rules and guidelines
/about - Bot information and features
/status - Your usage statistics and preferences
/reset - Reset all your data and preferences
/ping - Check bot responsiveness
/version - Bot version and system info
/feedback <message> - Send feedback to developers
/limit - View current rate limits
/privacy - Privacy policy and data handling
/export - Export your conversation history

**🔑 API Key Commands:**
/setkey <key> - Set your own Gemini API key
/keystatus - Check API key status
/removekey - Remove custom API key

**🖼️ Smart Capabilities:**
📸 **Photo Generation** - Send any photo for automatic transformation
📷 **Multiple Images** - Send multiple photos for combined Gemini processing
🎨 **Auto Generation** - Automatic image/text based on any input
💾 **Auto Storage** - All content saved locally
🤖 **Smart Response** - AI decides text or image automatically

**💡 Smart Features:**
🧠 **Context Memory** - Remembers 20 recent messages
👤 **User Profiles** - Persistent preferences
🎯 **Task Detection** - Auto-detects image requests
📁 **File Organization** - User-specific storage
🔄 **Multi-format** - JPEG, PNG, GIF, BMP, WebP support
📝 **Enhanced Typography** - Better text formatting with markdown

**🌐 REST API:** Available on port 5001
• GET /api/health - Health check
• POST /api/analyze-image - Analyze via JSON
• POST /api/analyze-image-file - File upload analysis
• POST /api/generate-image - Generation guidance
• GET /api/user-stats/<id> - Storage statistics

**📂 Your Storage:** `responses/user_{message.from_user.id}/`
├── images/ - Your uploaded images
└── analysis/ - AI analysis results

Ready to help with text conversations and image analysis! 🚀"""

        bot.reply_to(message, config_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in config command: {e}")
        bot.reply_to(message, "Sorry, I couldn't show the configuration. Please try again.")

@bot.message_handler(commands=['generate'])
@with_rate_limit
def handle_generate_command(message):
    """Generate an image using Gemini"""
    try:
        user_id = message.from_user.id
        
        # Extract prompt from command
        parts = message.text.split(' ', 1)
        if len(parts) < 2:
            bot.reply_to(message, "Please provide a description for the image you want to generate.\nUsage: /generate <description>")
            return
        
        prompt = parts[1]
        
        # Show typing indicator
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Generate image
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        image_path, text_response = loop.run_until_complete(
            agentic_bot.generate_image_with_gemini(user_id, prompt)
        )
        
        if image_path:
            # Send the generated image
            with open(image_path, 'rb') as img_file:
                bot.send_photo(
                    message.chat.id, 
                    img_file, 
                    caption=f"🎨 Generated image for: {prompt}"
                )
            
            # Clean up temporary file
            try:
                os.unlink(image_path)
            except:
                pass
                
        else:
            # Send text response if no image was generated
            formatted_response = agentic_bot.format_text_response(text_response, "explanation", user_id)
            bot.reply_to(message, formatted_response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in generate command: {e}")
        bot.reply_to(message, "Sorry, I couldn't generate the image. Please try again.")

@bot.message_handler(commands=['rules'])
def handle_rules(message):
    """Display bot rules and guidelines"""
    bot.reply_to(message, """
📋 **Bot Rules & Guidelines:**

✅ **Allowed:**
• Send text messages for conversation
• Upload images for analysis (max 10MB)
• Request image generation with descriptions
• Use bot commands appropriately
• Save conversation history

🚫 **Not Allowed:**
• Spam or excessive requests (max 10/min)
• Inappropriate or offensive content
• Attempting to exploit the bot
• Sharing malicious files

📊 **Current Limits:**
• 10 requests per minute per user
• 100 requests per hour per user
• 10MB maximum image file size
• 4000 characters maximum message length

💾 **Data Storage:**
• Analysis and activity logged in memory
• Conversation history (last 20 messages)
• User preferences stored
• Use /clear to reset your data

Use /help for more commands and features!
    """)

@bot.message_handler(commands=['about'])
def handle_about(message):
    """Display information about the bot"""
    bot.reply_to(message, """
🤖 **AI Assistant Bot**

**Powered by:** Gemini 2.0 Flash AI
**Features:**
• 🖼️ Image analysis and generation
• 💬 Smart conversation with memory
• 📁 Local data storage
• 🎨 Multiple image processing
• 🌐 REST API integration

**Version:** 1.0.0
**Storage:** Local filesystem
**API:** Available on port 5001

Created to help with image analysis, generation, and intelligent conversation!
    """)

@bot.message_handler(commands=['status'])
def handle_status(message):
    """Display user's current status and statistics"""
    user_id = message.from_user.id
    stats = agentic_bot.get_user_stats(user_id)
    context = agentic_bot.get_user_context(user_id)
    history_count = len(agentic_bot.get_conversation_history(user_id))
    
    bot.reply_to(message, f"""
📊 **Your Status:**

**📈 Usage Statistics:**
• Images processed: {stats.get('images_processed', 0)}
• Analysis count: {stats.get('analysis_count', 0)}
• Generated count: {stats.get('generated_count', 0)}
• Memory records: {stats.get('memory_records', 0)}
• Conversation messages: {history_count}

**⚙️ Current Preferences:**
• Text format: {context.get('text_format', 'enhanced')}
• Analysis style: {context.get('analysis_style', 'brief')}
• Next image mode: {context.get('next_image_mode', 'generate')}
• API key: {'✅ Custom' if agentic_bot.has_custom_api_key(user_id) else '🔄 Shared'}

**💾 Storage Type:**
{stats.get('storage_type', 'memory')} (no local files)

Use /setpref to change your preferences!
    """)

@bot.message_handler(commands=['reset'])
def handle_reset(message):
    """Reset all user data and preferences"""
    user_id = message.from_user.id
    
    # Clear conversation history
    agentic_bot.conversations[user_id] = []
    
    # Clear user context
    agentic_bot.user_contexts[user_id] = {}
    
    bot.reply_to(message, """
🔄 **All Data Reset Successfully!**

**Cleared:**
• Conversation history
• User preferences
• Context memory

**Note:** Analysis history in memory will be cleared on session timeout.
Use /stats to see your activity information.
    """)

@bot.message_handler(commands=['ping'])
def handle_ping(message):
    """Check if bot is responsive"""
    import time
    start_time = time.time()
    
    # Simple responsiveness test
    response_time = round((time.time() - start_time) * 1000, 2)
    
    bot.reply_to(message, f"""
🏓 **Pong!**

Bot is active and responsive.
Response time: {response_time}ms

**System Status:** ✅ Online
**AI Model:** Gemini 2.0 Flash
**Storage:** Memory-based (no files)
    """)

@bot.message_handler(commands=['version'])
def handle_version(message):
    """Display bot version and system information"""
    bot.reply_to(message, f"""
🔢 **Bot Version Information:**

**Bot Version:** 1.0.0
**Python Version:** {import_sys().version.split()[0] if 'import_sys' in globals() else 'Unknown'}
**Gemini Model:** gemini-2.0-flash-exp
**Advanced GenAI:** {'✅ Available' if ADVANCED_GENAI_AVAILABLE else '❌ Not available'}

**Last Updated:** 2024
**Features:** Image analysis, generation, conversation, memory storage
    """)

def import_sys():
    import sys
    return sys

@bot.message_handler(commands=['feedback'])
def handle_feedback(message):
    """Handle user feedback"""
    parts = message.text.split(' ', 1)
    if len(parts) < 2:
        bot.reply_to(message, """
💬 **Send Your Feedback:**

Usage: `/feedback Your message here`

Example:
`/feedback The image analysis is very accurate!`
`/feedback Could you add more image formats?`

Your feedback helps improve the bot! 🚀
        """)
    else:
        feedback_text = parts[1]
        user_id = message.from_user.id
        username = message.from_user.username or "Unknown"
        
        # Log feedback (in production, you might save to database)
        logger.info(f"Feedback from user {user_id} (@{username}): {feedback_text}")
        
        bot.reply_to(message, """
✅ **Thank you for your feedback!**

Your message has been received and will help improve the bot.

**Feedback:** """ + f'"{feedback_text}"' + """

Keep the suggestions coming! 🙏
        """)

@bot.message_handler(commands=['limit'])
def handle_limit(message):
    """Display current rate limits and usage policies"""
    bot.reply_to(message, """
⚡ **Current Rate Limits:**

**Per Minute:**
• 10 requests maximum
• Applies to all commands and messages

**Per Hour:**
• 100 requests maximum
• Resets every hour

**File Limits:**
• 10MB maximum image size
• Supported: PNG, JPG, JPEG, GIF, WebP
• 4000 characters max message length

**Memory Limits:**
• 20 messages conversation history
• Automatic cleanup of old data

**Note:** These limits ensure fair usage for all users.
    """)

@bot.message_handler(commands=['privacy'])
def handle_privacy(message):
    """Display privacy policy and data handling information"""
    bot.reply_to(message, """
🔒 **Privacy Policy:**

**Data Storage:**
• Conversations stored temporarily in memory (last 20 messages)
• Analysis results logged in memory
• User preferences stored in memory
• No local files created (avoids permission issues)

**Data Usage:**
• Only used to provide bot services
• Not shared with third parties
• No data sent to external services (except Gemini AI)
• Memory-based storage only

**Your Control:**
• Use `/clear` to delete conversation history
• Use `/reset` to clear all preferences
• Memory data cleared when session expires

**Security:**
• All data stored in memory only
• No persistent local files created
• Rate limiting prevents abuse

Questions? Use `/feedback` to ask!
    """)

@bot.message_handler(commands=['export'])
def handle_export(message):
    """Export user's conversation history"""
    user_id = message.from_user.id
    history = agentic_bot.get_conversation_history(user_id)
    
    if not history:
        bot.reply_to(message, "📄 No conversation history to export. Start chatting to build history!")
        return
    
    # Create export text
    export_text = "📄 **Conversation Export:**\n\n"
    
    for i, msg in enumerate(history[-10:], 1):  # Last 10 messages
        role = "You" if msg["role"] == "user" else "Bot"
        timestamp = msg.get("timestamp", 0)
        
        # Format timestamp
        if timestamp:
            import datetime
            dt = datetime.datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%m/%d %H:%M")
        else:
            time_str = "Unknown"
        
        # Truncate long messages
        content = msg["content"]
        if len(content) > 100:
            content = content[:100] + "..."
        
        export_text += f"**{i}. {role}** ({time_str}):\n{content}\n\n"
    
    export_text += f"**Total messages:** {len(history)}\n"
    export_text += "**Note:** Only last 10 messages shown. Use `/stats` for full statistics."
    
    bot.reply_to(message, export_text, parse_mode='Markdown')

@bot.message_handler(commands=['setkey'])
@with_rate_limit
def handle_set_api_key(message):
    """Set user's custom Gemini API key"""
    user_id = message.from_user.id
    
    try:
        # Extract API key from command
        parts = message.text.split(' ', 1)
        if len(parts) < 2:
            bot.reply_to(message, """
🔑 **Set Your Custom Gemini API Key**

Usage: `/setkey YOUR_API_KEY_HERE`

**Why use your own API key?**
• Use your own Gemini API quota
• More control over usage limits  
• Potentially better performance
• Your own billing and usage tracking

**How to get a Gemini API key:**
1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Create a new API key
4. Copy the key and use `/setkey YOUR_KEY`

**Security:** Your key is stored securely and only used for your requests.

Example: `/setkey AIzaSyC8B...`
            """)
            return
        
        api_key = parts[1].strip()
        
        # Validate the API key format
        if not api_key.startswith('AIza') or len(api_key) < 20:
            bot.reply_to(message, """
❌ **Invalid API Key Format**

Your API key should:
• Start with "AIza"
• Be at least 20 characters long
• Look like: AIzaSyC8B...

Please get a valid key from: https://aistudio.google.com/app/apikey
            """)
            return
        
        # Show processing message
        processing_msg = bot.reply_to(message, "🔄 **Testing your API key...**\nThis may take a few seconds...")
        
        # Test and set the API key
        success = agentic_bot.set_user_api_key(user_id, api_key)
        
        if success:
            # Delete the message containing the API key for security
            try:
                bot.delete_message(message.chat.id, message.message_id)
            except:
                pass
            
            # Update processing message with success
            bot.edit_message_text(
                """✅ **API Key Set Successfully!**

🔑 Your custom Gemini API key is now active
🚀 All AI features will now use your personal quota
📊 Use `/keystatus` to check your key status
🗑️ Use `/removekey` to remove your key

**Security Note:** Your original message has been deleted for privacy.
                """,
                processing_msg.chat.id,
                processing_msg.message_id,
                parse_mode='Markdown'
            )
        else:
            # Delete the message containing the API key for security
            try:
                bot.delete_message(message.chat.id, message.message_id)
            except:
                pass
            
            # Update processing message with error
            bot.edit_message_text(
                """❌ **API Key Validation Failed**

The provided API key could not be validated. Please check:

• Key format is correct (starts with AIza...)
• Key is active and valid
• You have Gemini API access enabled
• No extra spaces or characters

Get a valid key from: https://aistudio.google.com/app/apikey

**Security Note:** Your original message has been deleted for privacy.
                """,
                processing_msg.chat.id,
                processing_msg.message_id,
                parse_mode='Markdown'
            )
        
    except Exception as e:
        logger.error(f"Error setting API key for user {user_id}: {e}")
        # Try to delete the original message for security
        try:
            bot.delete_message(message.chat.id, message.message_id)
        except:
            pass
        bot.reply_to(message, "❌ An error occurred while setting your API key. Please try again.")

@bot.message_handler(commands=['keystatus'])
@with_rate_limit
def handle_key_status(message):
    """Check user's API key status"""
    user_id = message.from_user.id
    
    has_custom_key = agentic_bot.has_custom_api_key(user_id)
    
    if has_custom_key:
        api_key = agentic_bot.get_user_api_key(user_id)
        masked_key = f"{api_key[:10]}...{api_key[-4:]}" if len(api_key) > 14 else "Set"
        
        bot.reply_to(message, f"""
🔑 **Your API Key Status**

**Status:** ✅ Custom key active
**Key:** {masked_key}
**Usage:** Your personal Gemini quota

**Commands:**
• `/setkey <new_key>` - Update your key
• `/removekey` - Remove your key and use default
• `/limit` - View current usage limits

All AI features are using your personal API key!
        """)
    else:
        bot.reply_to(message, """
🔑 **Your API Key Status**

**Status:** 🔄 Using default shared key
**Usage:** Shared bot quota with rate limits

**To use your own key:**
1. Get API key: https://aistudio.google.com/app/apikey
2. Set it: `/setkey YOUR_API_KEY`

**Benefits of your own key:**
• Personal quota (no sharing)
• Better performance
• Your own usage tracking
        """)

@bot.message_handler(commands=['botstatus'])
@with_rate_limit  
def handle_bot_status(message):
    """Show multi-user bot status"""
    try:
        status = get_bot_status()
        
        # Get memory usage (optional)
        memory_info = "N/A"
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_info = f"{memory_mb:.1f}MB / {MAX_MEMORY_USAGE_MB}MB"
        except ImportError:
            memory_info = "N/A (psutil not available)"
        
        status_text = f"""🤖 **Bot Multi-User Status**

**👥 User Activity:**
• Active users: {status['active_users']}
• Total users: {status['total_users']}
• Max concurrent: {status['max_concurrent']}

**💾 System Resources:**
• Memory usage: {memory_info}
• Thread pool: 20 workers
• Session timeout: {USER_SESSION_TIMEOUT_MINUTES} minutes

**⚡ Features:**
✅ Rate limiting (10 requests/min per user)
✅ Thread-safe operations
✅ Auto cleanup of inactive users
✅ Concurrent image processing
✅ Per-user API keys support

**🔄 Performance:**
• File operations: Atomic writes with locking
• Memory management: Auto garbage collection
• Session management: Background cleanup
• Media groups: Batch processing support
        """
        
        bot.reply_to(message, status_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        bot.reply_to(message, "❌ Error getting bot status.")

@bot.message_handler(commands=['removekey'])
@with_rate_limit
def handle_remove_key(message):
    """Remove user's custom API key"""
    user_id = message.from_user.id
    
    if agentic_bot.has_custom_api_key(user_id):
        agentic_bot.remove_user_api_key(user_id)
        bot.reply_to(message, """
🗑️ **API Key Removed Successfully**

Your custom API key has been removed.

**Status:** Now using default shared bot key
**Effect:** You'll share quota with other users
**Rate Limits:** Standard bot limits apply

To set a new key: `/setkey YOUR_API_KEY`
        """)
    else:
        bot.reply_to(message, """
ℹ️ **No Custom Key to Remove**

You're already using the default shared bot key.

To set your own key: `/setkey YOUR_API_KEY`
Get a key from: https://aistudio.google.com/app/apikey
        """)

@bot.message_handler(func=lambda message: True, content_types=['text'])
@with_rate_limit
def handle_message(message):
    user_id = message.from_user.id
    user_message = message.text
    
    # Show typing indicator
    bot.send_chat_action(message.chat.id, 'typing')
    
    try:
        # Check if this is an image analysis request (without an actual image)
        analysis_keywords = [
            "analyze image", "describe image", "what's in the image", "image analysis",
            "analyze the image", "describe the image", "what is in the image",
            "tell me about the image", "explain the image", "image description",
            "what do you see in", "identify the image", "examine the image",
            "analyze this image", "describe this image", "what's in this image",
            "can you analyze", "can you describe", "please analyze", "please describe",
            "look at this image", "examine this photo", "tell me about this photo",
            "what does this image show", "what's happening in", "identify this",
            "read this image", "ocr", "text in image", "extract text"
        ]
        
        user_message_lower = user_message.lower()
        is_analysis_request = any(keyword in user_message_lower for keyword in analysis_keywords)
        
        if is_analysis_request:
            # This is a request for image analysis but no image was provided
            bot.reply_to(
                message, 
                "📸 I'd be happy to analyze an image for you! Please:\n\n"
                "1. Send me a photo directly, or\n"
                "2. Use `/analyze` then send a photo, or\n" 
                "3. Add analysis keywords (analyze, describe, etc.) in your photo caption\n\n"
                "💡 I can analyze any image you send and provide descriptions!\n\n"
                "🔧 **Analysis Styles:**\n"
                "• Brief (default): Concise 1-3 sentence descriptions\n"
                "• Detailed: Comprehensive analysis\n"
                "• Change with: `/setpref analysis_style brief` or `detailed`"
            )
            return
        
        # Try advanced generation first (auto text/image)
        image_path, text_response = agentic_bot.generate_advanced_gemini(user_message, user_id=user_id)
        
        if image_path:
            # Image was generated - send it
            with open(image_path, 'rb') as img_file:
                bot.send_photo(
                    message.chat.id, 
                    img_file, 
                    caption=f"🎨 Generated for: {user_message}"
                )
            
            # Clean up temporary file
            try:
                os.unlink(image_path)
            except:
                pass
        else:
            # No image generated, send text response or fallback to regular response
            if text_response and text_response.strip():
                # Add to conversation history
                agentic_bot.add_to_conversation(user_id, "user", user_message)
                agentic_bot.add_to_conversation(user_id, "assistant", text_response)
                # Format and send response
                formatted_response = agentic_bot.format_text_response(text_response, "explanation", user_id)
                bot.reply_to(message, formatted_response, parse_mode='Markdown')
            else:
                # Fallback to regular conversation response
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(agentic_bot.generate_response(user_id, user_message))
                bot.reply_to(message, response, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        # Fallback to regular response on error
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(agentic_bot.generate_response(user_id, user_message))
            bot.reply_to(message, response, parse_mode='Markdown')
        except:
            bot.reply_to(message, "I'm sorry, I encountered an error processing your message. Please try again.")

@bot.message_handler(content_types=['photo'])
@with_rate_limit
def handle_photo(message):
    user_id = message.from_user.id
    
    try:
        # Get the largest photo size
        photo = message.photo[-1]
        file_info = bot.get_file(photo.file_id)
        
        # Download the image
        downloaded_file = bot.download_file(file_info.file_path)
        image = Image.open(io.BytesIO(downloaded_file))
        
        # Check if this photo is part of a media group (multiple images)
        media_group_id = getattr(message, 'media_group_id', None)
        
        if media_group_id:
            # This is part of a multiple image upload
            logger.info(f"Photo is part of media group: {media_group_id}")
            
            # Add to media group collection
            agentic_bot.add_to_media_group(media_group_id, user_id, message, image, message.caption)
            
            # Cancel existing timer if any
            if media_group_id in agentic_bot.media_group_timers:
                agentic_bot.media_group_timers[media_group_id].cancel()
            
            # Set timer to process the group (wait 2 seconds for more images)
            timer = Timer(2.0, agentic_bot.process_media_group, args=[media_group_id])
            agentic_bot.media_group_timers[media_group_id] = timer
            timer.start()
            
            return  # Don't process single image when it's part of a group
        
        # Single image processing - Generate/Edit only
        # Show upload indicator for image generation
        bot.send_chat_action(message.chat.id, 'upload_photo')
        
        # Get prompt from caption or use default
        if message.caption:
            prompt = message.caption
        else:
            prompt = "Transform this image into a more realistic version, enhance the details and make it look like a real photograph"
        
        # Save reference image temporarily for advanced generation
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_file_path = tmp.name
            image.save(temp_file_path)
        
        # Try advanced generation with the uploaded image as reference
        image_path, text_response = agentic_bot.generate_advanced_gemini(
            text=prompt,
            file_name=temp_file_path,
            user_id=user_id
        )
        
        # Clean up temporary reference file
        try:
            os.unlink(temp_file_path)
        except:
            pass
        
        if image_path:
            # Send the generated image
            with open(image_path, 'rb') as img_file:
                bot.send_photo(
                    message.chat.id, 
                    img_file, 
                    caption=f"🎨 Generated from your image!\n\n💾 Generation logged in memory!"
                )
            
            # Log the generation in memory
            try:
                # Save generation record in memory
                agentic_bot.save_analysis_in_memory(
                    user_id, 
                    f"Generated from image: {prompt}", 
                    f"Successfully generated image using reference image. Caption: {prompt}",
                    "generation"
                )
                
                # Update user stats
                agentic_bot.update_user_stats(user_id, "image_processed")
                
            except Exception as e:
                logger.error(f"Error logging generation: {e}")
            
            # Clean up temporary generated file
            try:
                os.unlink(image_path)
            except:
                pass
        else:
            # If generation fails, send a helpful message
            bot.reply_to(message, "❌ I couldn't generate a new image from your photo. Please try:\n\n• Adding a specific description as caption\n• Using the /generate command with text\n• Checking if your API key is working with /keystatus")
        
    except Exception as e:
        import traceback
        logger.error(f"Error handling photo for user {user_id}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        bot.reply_to(message, f"I'm sorry, I encountered an error while processing your image. Please try again.\n\nError: {str(e)[:100]}...")

@bot.message_handler(content_types=['document', 'audio', 'video', 'voice'])
def handle_other_media(message):
    media_type = message.content_type
    bot.reply_to(message, f"I received a {media_type}! Currently, I can only process images. If this is an image file, please send it as a photo instead.")

# Error handler
@bot.middleware_handler(update_types=['message'])
def error_handler(bot_instance, message):
    try:
        return message
    except Exception as e:
        logger.error(f"Middleware error: {e}")
        return message

# REST API endpoints
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "AI Image Generation API is running"})

@app.route('/api/status', methods=['GET'])
def bot_status():
    """Multi-user bot status endpoint"""
    try:
        status = get_bot_status()
        
        # Get memory usage (optional)
        memory_mb = None
        try:
            import psutil
            process = psutil.Process()
            memory_mb = round(process.memory_info().rss / 1024 / 1024, 1)
        except ImportError:
            memory_mb = "N/A"
        
        # Get media group count
        with agentic_bot.media_groups_lock:
            media_groups_count = len(agentic_bot.media_groups)
        
        return jsonify({
            "status": "healthy",
            "multi_user": {
                "active_users": status["active_users"],
                "total_users": status["total_users"], 
                "max_concurrent": status["max_concurrent"],
                "media_groups_processing": media_groups_count
            },
            "system": {
                "memory_usage_mb": memory_mb,
                "max_memory_mb": MAX_MEMORY_USAGE_MB,
                "thread_pool_workers": 20,
                "session_timeout_minutes": USER_SESSION_TIMEOUT_MINUTES
            },
            "features": {
                "rate_limiting": True,
                "thread_safety": True,
                "auto_cleanup": True,
                "concurrent_processing": True
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/api/user-stats/<int:user_id>', methods=['GET'])
def get_user_stats_api(user_id):
    """Get user storage statistics via REST API"""
    try:
        stats = agentic_bot.get_user_stats(user_id)
        
        if "error" in stats:
            return jsonify({"error": stats["error"]}), 500
        
        return jsonify({
            "success": True,
            "user_stats": stats
        })
        
    except Exception as e:
        logger.error(f"Error in get_user_stats_api: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-image', methods=['POST'])
def generate_image_api():
    """Generate image via REST API"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Get required parameters
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Get optional parameters
        user_id = data.get('user_id', 0)
        reference_image_data = data.get('reference_image')
        reference_image_url = data.get('reference_image_url')
        
        # Process reference image if provided
        reference_image = None
        if reference_image_url:
            reference_image = agentic_bot.download_image_from_url(reference_image_url)
        elif reference_image_data:
            reference_image = agentic_bot.process_base64_image(reference_image_data)
        
        # Generate image
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        image_path, text_response = loop.run_until_complete(
            agentic_bot.generate_image_with_gemini(user_id, prompt, reference_image)
        )
        
        if image_path:
            # Convert generated image to base64 for response
            with open(image_path, 'rb') as img_file:
                import base64
                image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Clean up temporary file
            try:
                os.unlink(image_path)
            except:
                pass
            
            return jsonify({
                "success": True,
                "generated_image": f"data:image/png;base64,{image_base64}",
                "prompt": prompt,
                "text_response": text_response,
                "has_reference": reference_image is not None
            })
        else:
            return jsonify({
                "success": False,
                "error": "Image generation not available with current API version",
                "text_response": text_response,
                "prompt": prompt,
                "has_reference": reference_image is not None,
                "note": "Image generation requires the latest Gemini API. Text analysis and enhanced prompts are available."
            })
        
    except Exception as e:
        logger.error(f"Error in generate_image_api: {e}")
        return jsonify({"error": str(e)}), 500

def run_flask_app():
    """Run Flask app in a separate thread"""
    app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == '__main__':
    print("="*80)
    print("🤖 MULTI-USER AGENTIC GEMINI TELEGRAM BOT")
    print("="*80)
    
    print("\n📋 CONFIGURATION:")
    print(f"- Bot Token: {'✅ Set' if BOT_TOKEN != 'YOUR_BOT_TOKEN_HERE' else '❌ Not set'}")
    print(f"- Gemini API Key: {'✅ Set' if GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE' else '❌ Not set'}")
    print(f"- Advanced GenAI: {'✅ Available' if ADVANCED_GENAI_AVAILABLE else '⚠️  Fallback mode'}")
    print(f"- REST API Port: 5001")
    print(f"- Storage Type: Memory-based (no local files)")
    
    print("\n👥 MULTI-USER CONFIGURATION:")
    print(f"- Max Concurrent Users: {MAX_CONCURRENT_USERS}")
    print(f"- Max Conversations/User: {MAX_CONVERSATIONS_PER_USER}")
    print(f"- Session Timeout: {USER_SESSION_TIMEOUT_MINUTES} minutes")
    print(f"- Memory Limit: {MAX_MEMORY_USAGE_MB}MB")
    print(f"- Thread Pool Workers: 20")
    print(f"- Rate Limiting: 10 requests/min per user")
    
    if BOT_TOKEN == 'YOUR_BOT_TOKEN_HERE' or GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY_HERE':
        print("\n❌ CONFIGURATION ERROR:")
        print("Please set your environment variables:")
        print("export BOT_TOKEN='your_telegram_bot_token'")
        print("export GEMINI_API_KEY='your_gemini_api_key'")
        exit(1)
    
    print("\n📱 TELEGRAM BOT COMMANDS:")
    commands_info = [
        ("/start", "Initialize bot and get welcome message"),
        ("/help", "Show detailed help and all features"),
        ("/clear", "Clear conversation history"),
        ("/context", "View current user context and preferences"),
        ("/setpref <key> <value>", "Set user preferences"),
        ("/stats", "View local storage statistics"),
        ("/demo", "Demo image generation"),
        ("/generate <prompt>", "Generate images from text prompts"),

        ("/multitest", "Test multiple image functionality")
    ]
    
    for cmd, desc in commands_info:
        print(f"  {cmd:<25} - {desc}")
    
    print("\n🖼️ IMAGE CAPABILITIES:")
    image_features = [
        "📸 Photo Generation", "Send any photo for automatic transformation",
        "📷 Multiple Images", "Send multiple photos for combined processing",
        "🎨 Auto Generation", "Automatic image/text based on any input",
        "💾 Auto Storage", "All analysis and responses logged in memory",
        "🖼️ Smart Response", "AI decides text or image automatically"
    ]
    
    for i in range(0, len(image_features), 2):
        print(f"  {image_features[i]:<20} - {image_features[i+1]}")
    
    print("\n🌐 REST API ENDPOINTS:")
    api_endpoints = [
        ("GET  /api/health", "Health check and status"),
        ("GET  /api/status", "Multi-user bot status and performance"),
        ("POST /api/analyze-image", "Analyze image (JSON with base64/URL)"),
        ("POST /api/analyze-image-file", "Analyze image (file upload)"),
        ("POST /api/generate-image", "Image generation guidance"),
        ("GET  /api/user-stats/<id>", "User storage statistics")
    ]
    
    for endpoint, desc in api_endpoints:
        print(f"  {endpoint:<30} - {desc}")
    
    print("\n💡 MULTI-USER FEATURES:")
    smart_features = [
        "👥 Concurrent Users", f"Up to {MAX_CONCURRENT_USERS} users simultaneously",
        "⚡ Thread Safety", "Safe concurrent operations with locks",
        "🔄 Rate Limiting", "10 requests/min per user protection",
        "🧹 Auto Cleanup", "Memory management and session cleanup",
        "📊 Performance Monitor", "Real-time status via /botstatus command",
        "🔒 User Isolation", "Separate contexts, conversations & API keys"
    ]
    
    for i in range(0, len(smart_features), 2):
        print(f"  {smart_features[i]:<20} - {smart_features[i+1]}")
    
    print("\n📂 STORAGE STRUCTURE:")
    print("  Memory-based storage:")
    print("  ├── User conversations")
    print("  ├── Analysis history")
    print("  ├── Generation logs")
    print("  └── User statistics")
    print("  (No local files created - avoids permission issues)")
    
    try:
        # Start Flask API server in a separate thread
        print("\n🌐 Starting REST API server on http://localhost:5001")
        flask_thread = Thread(target=run_flask_app, daemon=True)
        flask_thread.start()
        
        print("🚀 Telegram Bot is running...")
        print("\n" + "="*80)
        print("✅ ALL SYSTEMS OPERATIONAL")
        
        if ADVANCED_GENAI_AVAILABLE:
            print("💬 Ready for auto text/image generation!")
            print("🎨 AI will automatically decide: text or image response")
        else:
            print("💬 Ready for intelligent text responses!")
            print("⚠️  Advanced image generation in fallback mode")
            print("🎨 Use /generate for explicit image generation attempts")
        
        print("📸 Send photos for automatic generation, multiple photos for combined processing")
        print("💬 Type anything for smart responses, use commands for specific functions") 
        print("🛑 Press CTRL+C to stop")
        print("="*80)
        
        bot.infinity_polling(timeout=10, long_polling_timeout=5)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}") 