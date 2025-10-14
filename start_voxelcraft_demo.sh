#!/bin/bash
# Quick start script for VoxelCraft AI Agents demo

echo "================================================"
echo "🎮 VoxelCraft AI Agents Demo"
echo "================================================"
echo ""
echo "This will:"
echo "1. Start the VoxelCraft server"
echo "2. Wait 3 seconds"
echo "3. Run the AI agents"
echo ""
echo "Press Ctrl+C to stop everything"
echo ""
echo "================================================"

# Start VoxelCraft server in background
echo "Starting VoxelCraft server..."
python run_voxelcraft.py &
SERVER_PID=$!

# Wait for server to start
sleep 3

echo ""
echo "Server started! Opening browser..."
echo "View at: http://127.0.0.1:8000/index.html?room=ai_agents&name=Observer"
echo ""

# Wait a bit more
sleep 2

echo "Starting AI agents..."
echo ""

# Run AI agents
python voxelcraft_ai_controller.py

# Cleanup: kill server when agents finish
echo ""
echo "AI agents finished! Stopping server..."
kill $SERVER_PID

echo "Done! 🎉"
