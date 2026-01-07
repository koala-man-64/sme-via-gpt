# Azure Container Apps (eastus2)

Target posture: **public internet**, **single replica**, **manual indexing**, **OpenAI API**, **stateless chat**, **Container App secrets**.

## Prereqs
- Azure CLI + Container Apps extension
- Logged in: `az login`

## Deploy (scripted)
This creates billable resources.

```bash
export RG="rg-sme-via-gpt"
export LOCATION="eastus2"
export APP_NAME="sme-via-gpt"
export ENV_NAME="sme-via-gpt-env"
export ACR_NAME="<globally-unique-acr-name>"
export IMAGE_TAG="v1"

export OPENAI_API_KEY="<set>"
export APP_PASSWORD="<set>"

export AZURE_STORAGE_CONNECTION_STRING="<set>"
export AZURE_STORAGE_PREFIX="optional/prefix/"

bash deploy/aca_deploy.sh
```

## Deploy (GitHub Actions)
The workflow in `.github/workflows/deploy-aca.yml` creates/updates resources and deploys on push to `main` (and can be run manually).

Configure these **GitHub Secrets**:
- `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` (recommended OIDC auth), or `AZURE_CREDENTIALS` (service principal JSON)
- `OPENAI_API_KEY`
- `APP_PASSWORD`
- `AZURE_STORAGE_CONNECTION_STRING`

Configure these **GitHub Variables**:
- `AZURE_LOCATION` (default: `eastus2`)
- `AZURE_RG`
- `AZURE_APP_NAME`
- `AZURE_CA_ENV_NAME`
- `AZURE_ACR_NAME`
- `AZURE_STORAGE_PREFIX` (optional)

## Post-deploy checks
```bash
FQDN="$(az containerapp show -n "$APP_NAME" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
curl -fsS "https://${FQDN}/healthz"
curl -fsS "https://${FQDN}/readyz" | cat
```

## Optional: persist the FAISS/cache
For single replica, mount an Azure Files share and set `RAG_CACHE_DIR` (e.g. `/mnt/rag_cache`) as an app env var.
