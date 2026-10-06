"""AWS Lambda handler — adapta FastAPI a Lambda via Mangum.

Se usa con container image (ECR) porque TensorFlow + Tesseract
exceden el límite de 250MB de deployment package.
"""
from mangum import Mangum

from app.main import app

# lifespan="auto" → ejecuta startup/shutdown events (warmup de modelo)
# Sin api_gateway_base_path: las rutas FastAPI ya incluyen /api/v1
handler = Mangum(app, lifespan="auto")
