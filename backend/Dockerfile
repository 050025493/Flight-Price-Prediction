# 1. Use Python 3.9 Slim (Smallest base image)
FROM python:3.9-slim

# 2. Set working directory
WORKDIR /app

# 3. Install CPU-Only PyTorch FIRST (To prevent GPU bloat)
# This forces it to look at the CPU-only warehouse
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 4. Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy your app code
COPY . .

# 6. Expose port
EXPOSE 8000

# 7. Start Server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]