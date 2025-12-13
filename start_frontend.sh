#!/bin/bash
# Start the Censorium frontend

cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Error: Dependencies not installed"
    echo "Please run: npm install"
    exit 1
fi

echo "Starting Censorium Frontend..."
echo "Frontend will be available at http://localhost:3000"
echo ""

# Start the development server
npm run dev




