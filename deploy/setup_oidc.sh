#!/bin/bash

# Configuration - REPLACE THESE WITH YOUR VALUES
GITHUB_ORG="koala-man-64"           # e.g., "my-org" or your username
GITHUB_REPO="sme-via-gpt"         # e.g., "sme-via-gpt"
RESOURCE_GROUP="rg-sme-via-gpt"   # e.g., "rg-sme-via-gpt"
LOCATION="eastus2"                       # e.g., "eastus2"
IDENTITY_NAME="id-sme-via-gpt-github"    # Name for your Managed Identity
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

echo "Using Subscription ID: $SUBSCRIPTION_ID"

# 1. Create Resource Group (if it doesn't exist)
echo "Creating User Assigned Managed Identity..."
az identity create --name "${IDENTITY_NAME}" --resource-group "${RESOURCE_GROUP}" --location "${LOCATION}"

# 2. Get the Client ID and Principal ID of the new Identity
CLIENT_ID=$(az identity show --name "${IDENTITY_NAME}" --resource-group "${RESOURCE_GROUP}" --query clientId -o tsv)
PRINCIPAL_ID=$(az identity show --name "${IDENTITY_NAME}" --resource-group "${RESOURCE_GROUP}" --query principalId -o tsv)

echo "Created Identity: ${IDENTITY_NAME}"
echo "Client ID: ${CLIENT_ID}"
echo "Principal ID: ${PRINCIPAL_ID}"

# 3. Create Federated Credential for GitHub Actions (Main Branch)
echo "Creating Federated Credential for 'main' branch..."
az identity federated-credential create \
  --name "github-actions-main" \
  --identity-name "${IDENTITY_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --issuer "https://token.actions.githubusercontent.com" \
  --subject "repo:${GITHUB_ORG}/${GITHUB_REPO}:ref:refs/heads/main" \
  --audience "api://AzureADTokenExchange"

# 4. Create Federated Credential for Pull Requests (Optional - useful for PR checks)
echo "Creating Federated Credential for 'pull_request'..."
az identity federated-credential create \
  --name "github-actions-pr" \
  --identity-name "${IDENTITY_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --issuer "https://token.actions.githubusercontent.com" \
  --subject "repo:${GITHUB_ORG}/${GITHUB_REPO}:pull_request" \
  --audience "api://AzureADTokenExchange"

# 5. Assign Permissions (Contributor on the Resource Group)
echo "Assigning 'Contributor' role to the Resource Group..."
az role assignment create \
  --assignee "${PRINCIPAL_ID}" \
  --role "Contributor" \
  --scope "/subscriptions/${SUBSCRIPTION_ID}/resourceGroups/${RESOURCE_GROUP}"

# 6. Assign Reader Permissions on Subscription (Required for Login validation)
echo "Assigning 'Reader' role to the Subscription (for login validation)..."
az role assignment create \
  --assignee "${PRINCIPAL_ID}" \
  --role "Reader" \
  --scope "/subscriptions/${SUBSCRIPTION_ID}"

echo "--------------------------------------------------"
echo "SETUP COMPLETE"
echo "--------------------------------------------------"
echo "Please set the following Secrets in GitHub:"
echo "AZURE_CLIENT_ID: $CLIENT_ID"
echo "AZURE_TENANT_ID: $(az account show --query tenantId -o tsv)"
echo "AZURE_SUBSCRIPTION_ID: $SUBSCRIPTION_ID"
echo "--------------------------------------------------"
