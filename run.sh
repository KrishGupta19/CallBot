#!/bin/bash
echo "Starting AI Voice Agent..."
echo "Make sure ngrok is running in another terminal: ngrok http 8765"
echo ""
echo "Make sure your .env file has all keys filled in."
echo ""
python server.py
