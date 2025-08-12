#!/bin/bash

# Set your repo URL here
REPO_URL="https://github.com/continuousactivelearning/agrichat-annam.git"
TARGET_DIR="agrichat-annam"

# Clone the repo
git clone "$REPO_URL"
cd "$TARGET_DIR" || exit 1

# Prompt for API keys (or you can hardcode them here)
read -p "Enter your GOOGLE_API_KEY: " GOOGLE_API_KEY
read -p "Enter your FIRECRAWL_API_KEY: " FIRECRAWL_API_KEY
read -p "Enter your HF_API_TOKEN: " HF_API_TOKEN

# Write to .env
cat > .env <<EOF
GOOGLE_API_KEY=$GOOGLE_API_KEY
FIRECRAWL_API_KEY=$FIRECRAWL_API_KEY
HF_API_TOKEN=$HF_API_TOKEN
EOF

echo ".env file created with your API keys."