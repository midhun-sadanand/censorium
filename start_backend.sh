#!/bin/bash
# Start the Censorium backend API

cd "$(dirname "$0")/backend"

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Error: Virtual environment not found"
    echo "Please run: cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if dependencies are installed
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Dependencies not installed"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

echo "Starting Censorium API..."
echo "API will be available at http://localhost:8000"
echo "API docs at http://localhost:8000/docs"
echo ""

# Start the server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload




