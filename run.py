"""PhisNet entrypoint script.

Run the web application:
    python run.py
    python run.py --port 8080
"""
import argparse
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the PhisNet web platform.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"  PhisNet: Phishing URL Detection Platform")
    print(f"  Running at: http://{args.host}:{args.port}")
    print(f"  API Docs:   http://{args.host}:{args.port}/docs")
    print(f"=======================================================\n")

    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
