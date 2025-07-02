# 🐳 Docker Deployment Guide

## Quick Start

### 1. Prerequisites
- Docker and Docker Compose installed
- Telegram Bot Token from [@BotFather](https://t.me/botfather)
- Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 2. Environment Setup with Secrets

**Option A: Using .env file (Recommended for Development)**

Copy the example environment file and configure your secrets:

```bash
# Copy the template
cp env.example .env

# Edit the .env file with your actual secrets
nano .env
```

Example `.env` file:
```bash
# Required secrets
BOT_TOKEN=1234567890:ABCDEFGHIJKLMNOPQRSTUVWXYZ-1234567890
GEMINI_API_KEY=AIzaSyC8BsQ-abcdefghijklmnopqrstuvwxyz1234567890

# Optional configuration
MAX_CONCURRENT_USERS=100
MAX_CONVERSATIONS_PER_USER=20
USER_SESSION_TIMEOUT_MINUTES=60
MAX_MEMORY_USAGE_MB=500
```

**Option B: Using Docker Secrets (Recommended for Production)**

Create Docker secrets:
```bash
# Create secrets for production
echo "your_telegram_bot_token" | docker secret create bot_token -
echo "your_gemini_api_key" | docker secret create gemini_api_key -
```

**Option C: Using Environment Variables Directly**

```bash
# Export environment variables
export BOT_TOKEN="your_telegram_bot_token"
export GEMINI_API_KEY="your_gemini_api_key"
```

### 3. Deploy with Docker Compose (Recommended)

**Basic deployment with .env file:**
```bash
# Ensure .env file is configured (see step 2)
# Build and start the bot
docker-compose up -d

# View logs
docker-compose logs -f telegram-bot

# Stop the bot
docker-compose down
```

**Production deployment with Docker secrets:**
```bash
# First create secrets (see Option B above)
# Then uncomment secrets section in docker-compose.yml
# Deploy using swarm mode
docker swarm init
docker stack deploy -c docker-compose.yml telegram-bot-stack
```

### 4. Deploy with Docker (Manual)

**Development with environment variables:**
```bash
# Build the image
docker build -t agentic-gemini-bot .

# Run with environment variables
docker run -d \
  --name agentic-gemini-bot \
  -p 5001:5001 \
  -e BOT_TOKEN="your_telegram_bot_token" \
  -e GEMINI_API_KEY="your_gemini_api_key" \
  -e MAX_CONCURRENT_USERS=100 \
  -e MAX_MEMORY_USAGE_MB=500 \
  --restart unless-stopped \
  agentic-gemini-bot
```

**Production with secrets file:**
```bash
# Create secrets file
echo "BOT_TOKEN=your_bot_token" > secrets.env
echo "GEMINI_API_KEY=your_api_key" >> secrets.env

# Run with secrets file
docker run -d \
  --name agentic-gemini-bot \
  -p 5001:5001 \
  --env-file secrets.env \
  --restart unless-stopped \
  agentic-gemini-bot

# Remove secrets file for security
rm secrets.env
```

## 📁 File Structure

```
project/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Orchestration setup with secrets
├── .dockerignore           # Exclude unnecessary files
├── env.example             # Environment variables template
├── .env                    # Your actual secrets (create from env.example)
├── agentic_gemini_bot.py   # Main bot application
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── DEPLOYMENT.md          # This deployment guide

Note: No data/ directory needed - using memory-based storage!
```

## 🌐 Access Points

After deployment:

- **Telegram Bot**: Available on Telegram via your bot token
- **REST API**: `http://localhost:5001/api/`
- **Health Check**: `http://localhost:5001/api/health`

## 🔧 Configuration

### Environment Variables

**Required Secrets:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BOT_TOKEN` | ✅ Yes | - | Telegram bot token from @BotFather |
| `GEMINI_API_KEY` | ✅ Yes | - | Google Gemini API key from Google AI Studio |

**Optional Configuration:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MAX_CONCURRENT_USERS` | ❌ No | `100` | Maximum concurrent users allowed |
| `MAX_CONVERSATIONS_PER_USER` | ❌ No | `20` | Max conversation history per user |
| `USER_SESSION_TIMEOUT_MINUTES` | ❌ No | `60` | User session timeout in minutes |
| `MAX_MEMORY_USAGE_MB` | ❌ No | `500` | Maximum memory usage in MB |
| `FLASK_HOST` | ❌ No | `0.0.0.0` | Flask server host |
| `FLASK_PORT` | ❌ No | `5001` | Flask server port |

**Docker Resource Limits:**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MAX_MEMORY_LIMIT` | ❌ No | `1G` | Docker container memory limit |
| `MIN_MEMORY_LIMIT` | ❌ No | `256M` | Docker container memory reservation |
| `MAX_CPU_LIMIT` | ❌ No | `0.5` | Docker container CPU limit |
| `MIN_CPU_LIMIT` | ❌ No | `0.1` | Docker container CPU reservation |

### Storage

**Memory-Based Storage (No Volumes Needed):**
- All user conversations stored in memory
- Analysis results logged in memory  
- No persistent local files created
- Automatic cleanup on session timeout
- No permission issues with file system access

## 🚀 Production Deployment

### Docker Swarm / Kubernetes

For production environments, consider:

1. **Resource Limits**: Memory limit set to 1GB, CPU limit to 0.5 cores
2. **Health Checks**: Built-in health monitoring on `/api/health`
3. **Persistent Storage**: Mount volumes for data persistence
4. **Environment Variables**: Use secrets management for API keys
5. **Reverse Proxy**: Use nginx/traefik for SSL termination

### Example Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-gemini-bot
spec:
  replicas: 1
  selector:
    matchLabels:
      app: agentic-gemini-bot
  template:
    metadata:
      labels:
        app: agentic-gemini-bot
    spec:
      containers:
      - name: bot
        image: agentic-gemini-bot:latest
        ports:
        - containerPort: 5001
        env:
        - name: BOT_TOKEN
          valueFrom:
            secretKeyRef:
              name: bot-secrets
              key: bot-token
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: bot-secrets
              key: gemini-api-key
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
          requests:
            memory: "256Mi"
            cpu: "100m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 5001
          initialDelaySeconds: 30
          periodSeconds: 30
```

## 🛠️ Troubleshooting

### Common Issues

1. **Bot doesn't start**:
   ```bash
   # Check logs
   docker-compose logs telegram-bot
   
   # Verify environment variables
   docker-compose exec telegram-bot env | grep -E "(BOT_TOKEN|GEMINI_API_KEY)"
   ```

2. **API not accessible**:
   ```bash
   # Test health endpoint
   curl http://localhost:5001/api/health
   
   # Check port binding
   docker ps
   ```

3. **Docker build issues** (FIXED):
   - ✅ psutil compilation errors resolved
   - ✅ Memory monitoring now optional with fallbacks
   - ✅ No build dependencies required
   - If you still see build errors, ensure Docker has sufficient resources

4. **Memory monitoring shows "N/A"**:
   - This is normal - psutil is optional for lighter containers
   - Memory management still works via periodic cleanup
   - To enable detailed monitoring, uncomment psutil in requirements.txt

### Monitoring

```bash
# Container stats
docker stats agentic-gemini-bot

# Health check
curl http://localhost:5001/api/health

# Application logs
docker-compose logs -f --tail=100
```

## 🔄 Updates

To update the bot:

```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 📊 API Endpoints

Once deployed, the following endpoints are available:

- `GET /api/health` - Health check
- `POST /api/analyze-image` - Analyze image (JSON)
- `POST /api/analyze-image-file` - Analyze image (file upload)
- `GET /api/user-stats/<user_id>` - User statistics
- `POST /api/generate-image` - Image generation guidance

## 🎯 Bot Commands

Available Telegram commands:

- `/start` - Initialize bot
- `/help` - Show help
- `/setkey <key>` - Set personal Gemini API key
- `/status` - View your status
- `/generate <prompt>` - Generate images
- `/analyze` - Switch to analysis mode

## 🔐 Security Notes

### Secrets Management

**Development:**
- Never commit `.env` files to version control
- Add `.env` to your `.gitignore` file
- Use `env.example` as template for team sharing
- Rotate API keys regularly

**Production:**
- Use Docker secrets or Kubernetes secrets
- Never pass secrets as command line arguments
- Use `--env-file` with restricted file permissions (600)
- Consider using external secret management (HashiCorp Vault, AWS Secrets Manager)

### Container Security

- ✅ Non-root user already configured (`botuser`)
- ✅ Minimal base image (Python 3.11 slim)
- ✅ No persistent file storage (memory-based)
- ✅ Health checks enabled
- ✅ Resource limits configured
- ✅ Optional dependencies (psutil) with graceful fallbacks
- ✅ No compilation dependencies needed

### Runtime Security

- Monitor API usage and set appropriate rate limits
- Keep base images updated for security patches
- Use HTTPS with reverse proxy in production
- Implement logging and monitoring
- Regular security audits of dependencies

### Example Production Security Setup

```bash
# Create restrictive secrets file
touch secrets.env
chmod 600 secrets.env
echo "BOT_TOKEN=your_token" > secrets.env
echo "GEMINI_API_KEY=your_key" >> secrets.env

# Deploy with secrets
docker run -d \
  --name secure-bot \
  --env-file secrets.env \
  --user 1000:1000 \
  --read-only \
  --tmpfs /tmp \
  --security-opt no-new-privileges \
  -p 127.0.0.1:5001:5001 \
  agentic-gemini-bot

# Clean up secrets
shred -u secrets.env
``` 