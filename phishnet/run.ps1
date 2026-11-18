# Start the PhishNet platform at http://localhost:8000
Set-Location $PSScriptRoot
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
