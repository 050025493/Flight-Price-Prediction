# 1. Use Python 3.9 (Lightweight version)
FROM python:3.9-slim

# 2. Set the working directory inside the server
WORKDIR /app

# 3. Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy all your backend files (server.py, models, csv)
COPY . .

# 5. Expose the port
EXPOSE 8000

# 6. Start the server
# host 0.0.0.0 is vital for cloud deployment
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]