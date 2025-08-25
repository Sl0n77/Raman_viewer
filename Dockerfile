FROM python:3.11-slim

# SciPy требует libgfortran5
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgfortran5 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости — лучше кэш слоёв
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем код
COPY app.py ./

# Конфиг Streamlit
RUN mkdir -p /root/.streamlit && \
    printf "[server]\nheadless = true\nport = 8501\naddress = \"0.0.0.0\"\n" > /root/.streamlit/config.toml && \
    printf "[browser]\ngatherUsageStats = false\n" >> /root/.streamlit/config.toml

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
