FROM python:3.9-slim

WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5678

ENV NAME World

CMD ["python", "./main.py"]
