# @title Setup WhisperLiveKit with Cloudflare Tunnel (PCM Input Enabled)
# @markdown This cell installs dependencies, sets up the large-v3-turbo model, enables raw PCM input, and exposes the server via Cloudflare.

import subprocess
import time
import sys
import os

# 1. Install System Dependencies
print("Installing ffmpeg...")
!apt-get install -y ffmpeg > /dev/null 2>&1

# 2. Install WhisperLiveKit and Faster-Whisper
print("Installing WhisperLiveKit and dependencies...")
!pip install whisperlivekit faster-whisper > /dev/null 2>&1

# 3. Install Cloudflare Tunnel (cloudflared)
print("Installing Cloudflare Tunnel...")
!wget -q -O cloudflared-linux-amd64.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb > /dev/null 2>&1

# 4. Start WhisperLiveKit Server
# Added --pcm-input flag here
print("Starting WhisperLiveKit Server (Model: large-v3-turbo, PCM Input: Enabled)...")
log_file = open("wlk_server.log", "w")
server_process = subprocess.Popen(
    [
        "wlk",
        "--model", "large-v3-turbo",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--backend", "faster-whisper",
        "--pcm-input"  # <--- Added this flag
    ],
    stdout=log_file,
    stderr=subprocess.STDOUT
)

# 5. Wait for the server to start (polling port 8000)
print("Waiting for server to initialize...")
import socket
def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

max_retries = 30
for i in range(max_retries):
    if check_port(8000):
        print("Server is up and running!")
        break
    time.sleep(2)
else:
    print("Warning: Server execution is taking longer than expected. Checking logs...")

# 6. Start Cloudflare Tunnel
print("Starting Cloudflare Tunnel...")
tunnel_process = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# 7. Extract and Print the Public URL
print("Searching for your public URL...")
time.sleep(5) 
found_url = False
try:
    start_time = time.time()
    while time.time() - start_time < 20: 
        line = tunnel_process.stderr.readline()
        if not line:
            break
        if "trycloudflare.com" in line:
            import re
            url_match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
            if url_match:
                public_url = url_match.group(0)
                print(f"\n✅ \033[92mYour Public URL is: {public_url}\033[0m")
                print(f"Use this URL in your client.")
                found_url = True
                break
except KeyboardInterrupt:
    print("\nStopping services...")
finally:
    if not found_url:
        print("\nCould not auto-detect URL yet. Check the logs manually.")