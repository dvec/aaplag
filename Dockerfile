FROM python:3.6

WORKDIR /app/
ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . .
CMD ["python", "./run.py"]
