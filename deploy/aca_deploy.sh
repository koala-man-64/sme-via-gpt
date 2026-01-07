#!/usr/bin/env bash
set -euo pipefail

# Deploy to Azure Container Apps (public internet, single replica, manual indexing).
# Prereqs:
#   - az CLI logged in: az login
#   - set required env vars below before running
#
# Notes:
#   - This script creates billable resources (RG, Log Analytics, ACR, Container Apps env, Container App).
#   - Secrets are set as Container App secrets (not echoed).

: "${RG:?Set RG (resource group name)}"
: "${LOCATION:=eastus2}"
: "${APP_NAME:?Set APP_NAME (container app name)}"
: "${ENV_NAME:=${APP_NAME}-env}"
: "${ACR_NAME:?Set ACR_NAME (globally-unique ACR name)}"
: "${IMAGE_TAG:=v1}"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${APP_PASSWORD:?Set APP_PASSWORD}"

# Storage config (simplest: store connection string as a Container App secret)
: "${AZURE_STORAGE_CONNECTION_STRING:?Set AZURE_STORAGE_CONNECTION_STRING}"
: "${AZURE_STORAGE_PREFIX:=}"

LAW_NAME="${APP_NAME}-law"
STORAGE_CONNSTR_SECRET_NAME="storage-connstr" # Container Apps secret name max length is 20

az group create -n "${RG}" -l "${LOCATION}" 1>/dev/null

if ! az monitor log-analytics workspace show -g "${RG}" -n "${LAW_NAME}" 1>/dev/null 2>&1; then
  az monitor log-analytics workspace create -g "${RG}" -n "${LAW_NAME}" -l "${LOCATION}" 1>/dev/null
fi
LAW_ID="$(az monitor log-analytics workspace show -g "${RG}" -n "${LAW_NAME}" --query customerId -o tsv)"
LAW_KEY="$(az monitor log-analytics workspace get-shared-keys -g "${RG}" -n "${LAW_NAME}" --query primarySharedKey -o tsv)"

if ! az containerapp env show -n "${ENV_NAME}" -g "${RG}" 1>/dev/null 2>&1; then
  az containerapp env create -n "${ENV_NAME}" -g "${RG}" -l "${LOCATION}" \
    --logs-workspace-id "${LAW_ID}" --logs-workspace-key "${LAW_KEY}" 1>/dev/null
fi

if ! az acr show -n "${ACR_NAME}" -g "${RG}" 1>/dev/null 2>&1; then
  az acr create -n "${ACR_NAME}" -g "${RG}" --sku Basic --admin-enabled true 1>/dev/null
else
  az acr update -n "${ACR_NAME}" -g "${RG}" --admin-enabled true 1>/dev/null
fi
ACR_LOGIN_SERVER="$(az acr show -n "${ACR_NAME}" --query loginServer -o tsv)"
ACR_USERNAME="$(az acr credential show -n "${ACR_NAME}" --query username -o tsv)"
ACR_PASSWORD="$(az acr credential show -n "${ACR_NAME}" --query 'passwords[0].value' -o tsv)"

az acr build -r "${ACR_NAME}" -t "${APP_NAME}:${IMAGE_TAG}" . 1>/dev/null

if ! az containerapp show -n "${APP_NAME}" -g "${RG}" 1>/dev/null 2>&1; then
  az containerapp create -n "${APP_NAME}" -g "${RG}" --environment "${ENV_NAME}" \
    --image "${ACR_LOGIN_SERVER}/${APP_NAME}:${IMAGE_TAG}" \
    --ingress external --target-port 8000 \
    --min-replicas 1 --max-replicas 1 \
    --system-assigned \
    --registry-server "${ACR_LOGIN_SERVER}" \
    --registry-username "${ACR_USERNAME}" \
    --registry-password "${ACR_PASSWORD}" \
    --secrets \
      openai-api-key="${OPENAI_API_KEY}" \
      app-password="${APP_PASSWORD}" \
      "${STORAGE_CONNSTR_SECRET_NAME}=${AZURE_STORAGE_CONNECTION_STRING}" \
    --env-vars \
      OPENAI_API_KEY=secretref:openai-api-key \
      APP_PASSWORD=secretref:app-password \
      AUTH_REQUIRED=true \
      AZURE_STORAGE_CONNECTION_STRING="secretref:${STORAGE_CONNSTR_SECRET_NAME}" \
      AZURE_STORAGE_PREFIX="${AZURE_STORAGE_PREFIX}" 1>/dev/null
else
  az containerapp secret set -n "${APP_NAME}" -g "${RG}" --secrets \
    openai-api-key="${OPENAI_API_KEY}" \
    app-password="${APP_PASSWORD}" \
    "${STORAGE_CONNSTR_SECRET_NAME}=${AZURE_STORAGE_CONNECTION_STRING}" 1>/dev/null

  az containerapp update -n "${APP_NAME}" -g "${RG}" \
    --image "${ACR_LOGIN_SERVER}/${APP_NAME}:${IMAGE_TAG}" \
    --min-replicas 1 --max-replicas 1 \
    --set-env-vars \
      OPENAI_API_KEY=secretref:openai-api-key \
      APP_PASSWORD=secretref:app-password \
      AUTH_REQUIRED=true \
      AZURE_STORAGE_CONNECTION_STRING="secretref:${STORAGE_CONNSTR_SECRET_NAME}" \
      AZURE_STORAGE_PREFIX="${AZURE_STORAGE_PREFIX}" 1>/dev/null

  az containerapp ingress update -n "${APP_NAME}" -g "${RG}" --type external --target-port 8000 1>/dev/null
fi

echo "Deployed: $(az containerapp show -n "${APP_NAME}" -g "${RG}" --query properties.configuration.ingress.fqdn -o tsv)"
