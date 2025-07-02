
---
title: Gemini Telegram Bot
emoji: 👁
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


<a href="https://web.telegram.org/k/#@ameer1122334455_bot"><img src="https://img.shields.io/badge/Chat-on%20Telegram-2CA5E0.svg?logo=telegram&style=for-the-badge" alt="Chat on Telegram"/></a>


# 🤖 Agentic Gemini Bot with Image Analysis

An intelligent Telegram bot powered by Google Gemini that supports both text conversations and image analysis, with REST API endpoints for programmatic access.

## ✨ Features

### 🤖 Telegram Bot
- **🧠 Intelligent Responses**: Powered by Google Gemini Pro AI model
- **💭 Conversation Memory**: Remembers context within chat sessions
- **🖼️ Image Analysis**: Send photos to get AI-powered analysis
- **👤 User Personalization**: Stores user preferences and context
- **🔄 Contextual Awareness**: References previous messages and topics
- **📚 Multi-Topic Support**: Handles various types of questions and tasks

### 🌐 REST API
- **📡 Image Analysis from URLs**: Analyze images from web URLs
- **🔗 Base64 Image Processing**: Process base64-encoded images
- **📁 File Upload Support**: Direct image file uploads
- **💾 Automatic Storage**: Local saving of images and analysis results
- **📊 Storage Tracking**: User statistics and storage management
- **❤️ Health Monitoring**: API health check endpoint

### 🧠 AI Capabilities
- **📝 Text Generation**: Powered by Gemini 2.5 Pro
- **👁️ Vision Analysis**: Powered by Gemini 2.5 Pro Vision
- **🧠 Context Awareness**: Remembers conversation history
- **🎯 Custom Prompts**: Support for custom analysis prompts

## 🚀 Quick Start

### Prerequisites

1. **Telegram Bot Token**: Get one from [@BotFather](https://t.me/botfather)
2. **Google Gemini API Key**: Get one from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Installation

1. **Clone or download the files**:
   ```bash
   git clone <your-repo> # or download the files
   cd <project-directory>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   ```bash
   export BOT_TOKEN="your_telegram_bot_token_here"
   export GEMINI_API_KEY="your_gemini_api_key_here"
   ```

   Or create a `.env` file:
   ```
   BOT_TOKEN=your_telegram_bot_token_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

4. **Run the bot**:
   ```bash
   python agentic_gemini_bot.py
   ```
   
   This will start both the Telegram bot and REST API server on `http://localhost:5001`

## 🎯 Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Initialize the bot and get a welcome message |
| `/help` | Show detailed help and features |
| `/clear` | Clear conversation history |
| `/context` | View your current context and preferences |
| `/setpref` | Set user preferences (e.g., `/setpref language English`) |
| `/stats` | View your local storage statistics |

## 🖼️ Image Analysis

### Via Telegram Bot
1. Send any photo to the bot
2. Optionally add a caption with specific questions
3. Get detailed AI analysis of the image

### Via REST API

#### Health Check
```bash
GET http://localhost:5001/api/health
```

#### Analyze Image from URL
```bash
POST http://localhost:5001/api/analyze-image
Content-Type: application/json

{
  "image_url": "https://example.com/image.jpg",
  "prompt": "Describe this image in detail",
  "user_id": 123
}
```

#### Analyze Base64 Image
```bash
POST http://localhost:5001/api/analyze-image
Content-Type: application/json

{
  "image": "data:image/jpeg;base64,/9j/4AAQ...",
  "prompt": "What objects do you see?",
  "user_id": 456
}
```

#### Upload Image File
```bash
POST http://localhost:5001/api/analyze-image-file
Content-Type: multipart/form-data

FormData:
- image: [file]
- prompt: "Analyze this image"
- user_id: 789
```

#### Get User Statistics
```bash
GET http://localhost:5001/api/user-stats/123
```

## 💡 Usage Examples

### Basic Q&A
```
User: What is machine learning?
Bot: [Provides detailed explanation with context]

User: Can you give me an example?
Bot: [Provides examples while remembering the ML context]
```

### Contextual Conversations
```
User: I'm planning a trip to Japan
Bot: [Responds with travel advice]

User: What about the food there?
Bot: [Continues with Japanese cuisine info, remembering the travel context]
```

### Setting Preferences
```
User: /setpref language Spanish
Bot: ✅ Preference updated: language = Spanish

User: Hello
Bot: [Responds considering the Spanish language preference]
```

### Image Analysis Examples
```
User: [Sends a photo of a sunset]
Bot: 🖼️ Image Analysis: This is a beautiful sunset scene with vibrant orange and pink hues...

User: [Sends a photo with caption "What flowers are these?"]
Bot: 🖼️ Image Analysis: These appear to be roses in full bloom...

User: /stats
Bot: 📊 Your Storage Stats:
     🖼️ Images Saved: 5
     📝 Analysis Saved: 5
     📁 Storage Path: responses/user_123456
```

## 💾 Local Storage

All image analysis results are automatically saved locally with user organization:

### Directory Structure
```
responses/
├── user_123/
│   ├── images/          # Original images saved as PNG
│   └── analysis/        # Analysis results as JSON
├── user_456/
│   ├── images/
│   └── analysis/
└── ...
```

### File Naming Convention
- **Images**: `image_YYYYMMDD_HHMMSS_<prompt_hash>.png`
- **Analysis**: `analysis_YYYYMMDD_HHMMSS_<prompt_hash>.json`

### Analysis JSON Structure
```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "user_id": 123,
  "prompt": "What do you see in this image?",
  "response": "AI analysis response text...",
  "image_filename": "image_20240115_103045_abc12345.png",
  "metadata": {
    "response_length": 256,
    "word_count": 45
  }
}
```

### Storage Commands
- **Telegram**: `/stats` - View your storage statistics
- **API**: `GET /api/user-stats/<user_id>` - Get storage info programmatically

## 🧪 Testing

Run the included test scripts:

```bash
# Test all API endpoints
python test_image_api.py

# Test local storage functionality
python test_local_storage.py
```

## 📋 API Response Format

All image analysis endpoints return:

```json
{
  "success": true,
  "analysis": "Detailed AI analysis of the image...",
  "prompt": "The prompt used for analysis",
  "filename": "original_filename.jpg"
}
```

**Note**: All responses are automatically saved locally in the `responses/user_<id>/` directory.

Error responses:
```json
{
  "error": "Error description"
}
```

## 🔧 Technical Details

### Architecture
```
┌─────────────────┐    ┌──────────────────┐
│   Telegram Bot  │    │    REST API      │
│     (Thread)    │    │   (Flask App)    │
└─────────┬───────┘    └─────────┬────────┘
          │                      │
          └──────────┬───────────┘
                     │
            ┌────────▼─────────┐
            │   AgenticBot     │
            │   (Core Logic)   │
            └────────┬─────────┘
                     │
            ┌────────▼─────────┐
            │  Google Gemini   │
            │ (Text + Vision)  │
            └──────────────────┘
```

- **AgenticBot Class**: Main bot logic with conversation and image analysis
- **Conversation History**: Stores last 20 messages per user
- **User Context**: Persistent user preferences and information
- **Dual Models**: Text (Gemini 2.5 Pro) and Vision (Gemini 2.5 Pro Vision)

### Key Components
- **Gemini Integration**: Uses `google-generativeai` library for both text and vision
- **Flask API**: REST endpoints for programmatic access
- **Image Processing**: PIL-based image handling with multiple input formats
- **Threading**: Concurrent Telegram bot and API server operation
- **Memory Management**: Automatic conversation history cleanup
- **Error Handling**: Comprehensive error handling and logging

### Supported Image Formats
- JPEG, PNG, GIF, BMP, WebP
- Base64 encoded images
- URLs to images
- Direct file uploads

### Security Features
- Environment variable configuration
- Input validation and sanitization
- Error logging without exposing sensitive data
- Rate limiting through Telegram's built-in mechanisms

## 🛠️ Customization

### Modifying the System Prompt
Edit the `system_prompt` in the `AgenticBot` class to change the bot's personality:

```python
self.system_prompt = """Your custom system prompt here..."""
```

### Adding New Commands
Add new command handlers following the pattern:

```python
@bot.message_handler(commands=['newcommand'])
def handle_new_command(message):
    # Your command logic here
    pass
```

### Extending Context Management
Modify the `update_user_context` method to add custom context handling:

```python
def update_user_context(self, user_id: int, key: str, value: str):
    # Custom context logic
    pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Bot doesn't respond**:
   - Check if BOT_TOKEN is correct
   - Verify the bot is running
   - Check network connectivity

2. **Gemini API errors**:
   - Verify GEMINI_API_KEY is valid
   - Check API quotas and limits
   - Ensure API is enabled

3. **Memory issues**:
   - Bot automatically manages memory by keeping last 20 messages
   - Use `/clear` to reset conversation history

### Debug Mode
Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📋 Requirements

- Python 3.8+
- Telegram Bot Token
- Google Gemini API Key
- Internet connection

## 🔒 Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- Monitor API usage and costs
- Implement rate limiting for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Telegram Bot API documentation
3. Check Google Gemini API documentation
4. Create an issue in the repository

---

**Happy chatting with your intelligent AI assistant! 🎉** 
