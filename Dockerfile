# Placeholder — will be finalised once the API and model are ready.
# Must run with: docker run -p 8000:8000 -e API_KEY=<key> <image>
# Must NOT need internet at runtime, so the trained model is copied into the image.
FROM python:3.11-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY model/ model/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
