# Используем официальный Python-образ
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем зависимости
RUN pip install flask
RUN pip install ultralytics
RUN apt-get update && apt-get install libgl1 -y
RUN apt-get install libglib2.0-0 -y

COPY requirements.txt .
# Устанавливаем зависимости
RUN pip install -r requirements.txt

# Копируем файлы приложения
COPY app/ ./app

# Открываем порт для Flask
EXPOSE 5000

# Запуск приложения
CMD ["python", "app/app.py"]
