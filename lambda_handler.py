"""AWS Lambda handler — adapta FastAPI a Lambda via Mangum.

Se usa con container image (ECR) porque TensorFlow + Tesseract
exceden el límite de 250MB de deployment package.
"""
from mangum import Mangum

from app.main import app

# lifespan="auto" → ejecuta startup/shutdown events (warmup de modelo)
handler = Mangum(app, lifespan="auto", api_gateway_base_path="/api/v1")
