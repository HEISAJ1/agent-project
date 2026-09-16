# Week 4: Dockerfile — packages the whole app so it runs identically
# anywhere (your laptop, Render, any other host), instead of depending on
# whatever happens to be installed locally.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first, separately from the app code. Docker caches
# each step — as long as requirements.txt doesn't change, rebuilds after
# editing your actual code skip reinstalling everything from scratch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code.
COPY . .

# Render (and most hosts) set the port to listen on via the $PORT
# environment variable at runtime, rather than a fixed number — so the
# container needs to read that instead of hardcoding 8000.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]