FROM python:3.8
COPY requirements.txt /demo/requirements.txt
EXPOSE 5000
RUN pip install --no-cache-dir -r /demo/requirements.txt
