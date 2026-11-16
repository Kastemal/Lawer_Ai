FROM python:3.12-slim

WORKDIR /app

# Системные утилиты для сборки (если потребуются)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода и данных
COPY . .

# Порт веб-интерфейса
EXPOSE 8000

# Запуск веб-сервиса
CMD ["python", "web_app.py"]
