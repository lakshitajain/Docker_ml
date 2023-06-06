FROM python:3.8

COPY requirements.txt /demo/requirements.txt

COPY run.sh /demo/run.sh
EXPOSE 5000
RUN pip install --no-cache-dir -r requirements.txt
