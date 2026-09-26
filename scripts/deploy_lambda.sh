#!/usr/bin/env bash
# Deploy CollectorManager backend a AWS Lambda + API Gateway (container image)
#
# Requiere: aws cli configurado (deploy-user), Docker instalado
# Uso: ./scripts/deploy_lambda.sh [region]
set -euo pipefail

REGION="${1:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="collector-backend"
LAMBDA_NAME="collector-backend"
API_NAME="collector-backend-api"
IMAGE_TAG="latest"

echo "═══════════════════════════════════════════"
echo "  Deploy CollectorManager → Lambda"
echo "  Cuenta: $ACCOUNT_ID  Región: $REGION"
echo "═══════════════════════════════════════════"

# ── 1. Crear ECR repo si no existe ──────────────────────────────────────────
echo ""
echo "▶ [1/5] ECR Repository..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $REGION 2>/dev/null || \
  aws ecr create-repository --repository-name $ECR_REPO --region $REGION > /dev/null
echo "  ✅ $ECR_REPO"

# ── 2. Build + Push Docker image ────────────────────────────────────────────
echo ""
echo "▶ [2/5] Build + Push imagen Docker..."
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO"
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker build -f Dockerfile.lambda -t $ECR_REPO:$IMAGE_TAG .
docker tag $ECR_REPO:$IMAGE_TAG $ECR_URI:$IMAGE_TAG
docker push $ECR_URI:$IMAGE_TAG
echo "  ✅ Imagen subida: $ECR_URI:$IMAGE_TAG"

# ── 3. Crear Lambda function ────────────────────────────────────────────────
echo ""
echo "▶ [3/5] Lambda Function..."
if aws lambda get-function --function-name $LAMBDA_NAME --region $REGION 2>/dev/null; then
  echo "  Actualizando código..."
  aws lambda update-function-code \
    --function-name $LAMBDA_NAME \
    --image-uri $ECR_URI:$IMAGE_TAG \
    --region $REGION > /dev/null
else
  echo "  Creando función..."
  # Role mínimo para Lambda + MongoDB Atlas (solo necesita logs)
  ROLE_NAME="collector-lambda-role"
  if ! aws iam get-role --role-name $ROLE_NAME 2>/dev/null; then
    cat > /tmp/lambda-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "lambda.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
EOF
    aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file:///tmp/lambda-trust.json > /dev/null
    aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    sleep 5
  fi
  ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query Role.Arn --output text)

  aws lambda create-function \
    --function-name $LAMBDA_NAME \
    --package-type Image \
    --code ImageUri=$ECR_URI:$IMAGE_TAG \
    --role $ROLE_ARN \
    --timeout 30 \
    --memory-size 2048 \
    --region $REGION > /dev/null
fi
echo "  ✅ $LAMBDA_NAME"

# ── 4. API Gateway (HTTP API) ───────────────────────────────────────────────
echo ""
echo "▶ [4/5] API Gateway..."
API_ID=$(aws apigatewayv2 get-apis --region $REGION --query "Items[?Name=='$API_NAME'].ApiId" --output text)
if [ -z "$API_ID" ] || [ "$API_ID" = "None" ]; then
  API_ID=$(aws apigatewayv2 create-api \
    --name $API_NAME \
    --protocol-type HTTP \
    --region $REGION \
    --query ApiId --output text)
  echo "  API creada: $API_ID"
fi

# Integración Lambda
LAMBDA_ARN=$(aws lambda get-function --function-name $LAMBDA_NAME --region $REGION --query Configuration.FunctionArn --output text)
INTEGRATION_ID=$(aws apigatewayv2 create-integration \
  --api-id $API_ID \
  --integration-type AWS_PROXY \
  --integration-uri $LAMBDA_ARN \
  --payload-format-version 2.0 \
  --region $REGION \
  --query IntegrationId --output text 2>/dev/null || echo "exists")

# Ruta catch-all
aws apigatewayv2 create-route \
  --api-id $API_ID \
  --route-key "ANY /{proxy+}" \
  --target "integrations/$INTEGRATION_ID" \
  --region $REGION 2>/dev/null || true

aws apigatewayv2 create-route \
  --api-id $API_ID \
  --route-key "ANY /" \
  --target "integrations/$INTEGRATION_ID" \
  --region $REGION 2>/dev/null || true

# Stage default
aws apigatewayv2 create-stage \
  --api-id $API_ID \
  --stage-name '$default' \
  --auto-deploy \
  --region $REGION 2>/dev/null || true

# Permiso para que API Gateway invoque Lambda
aws lambda add-permission \
  --function-name $LAMBDA_NAME \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:$REGION:$ACCOUNT_ID:$API_ID/*" \
  --region $REGION 2>/dev/null || true

API_URL="https://$API_ID.execute-api.$REGION.amazonaws.com"
echo "  ✅ API Gateway: $API_URL"

# ── 5. Environment variables ────────────────────────────────────────────────
echo ""
echo "▶ [5/5] Variables de entorno..."
aws lambda update-function-configuration \
  --function-name $LAMBDA_NAME \
  --environment "Variables={
    MONGO_URL=mongodb+srv://cl0udd4t4dev_db_user:TxreXwqygHJWVevw@collector.x0qoxf3.mongodb.net/?appName=collector,
    MONGO_DB=collector_manager,
    JWT_SECRET=prod-secret-change-me-32bytes-minimum!!,
    ENV=prod
  }" \
  --region $REGION > /dev/null
echo "  ✅ Variables configuradas"

echo ""
echo "═══════════════════════════════════════════"
echo "  ✅ DEPLOY COMPLETADO"
echo "═══════════════════════════════════════════"
echo ""
echo "  API URL: $API_URL/api/v1"
echo "  Health:  $API_URL/health"
echo ""
echo "  Actualizar EXPO_PUBLIC_API_URL en la app:"
echo "  EXPO_PUBLIC_API_URL=$API_URL"
