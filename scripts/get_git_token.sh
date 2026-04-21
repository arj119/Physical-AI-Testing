#!/bin/bash
# Fetch a Foundry Git token for cloning/pushing the repo.
# Uses your personal browser token (from Foundry UI > Settings > Tokens).
#
# Usage:
#   source scripts/get_git_token.sh                          # prompts for token
#   PALANTIR_TOKEN=eyJ... source scripts/get_git_token.sh    # pass token directly

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load Foundry URL from .env
if [ -f "$PROJECT_DIR/.env" ]; then
    FOUNDRY_URL=$(grep '^FOUNDRY_URL=' "$PROJECT_DIR/.env" | cut -d= -f2-)
fi

REPO_RID="${REPO_RID:-ri.stemma.main.repository.ddb28091-90ea-49fb-b207-c9e55ecd1d8e}"

if [ -z "$PALANTIR_TOKEN" ]; then
    echo -n "Paste your Foundry personal token (from browser): "
    read -s PALANTIR_TOKEN
    echo ""
fi

echo "Requesting Git token from ${FOUNDRY_URL}..."

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${FOUNDRY_URL}/code/api/security/gitToken" \
    -H "accept: application/json" \
    -H "authorization: Bearer ${PALANTIR_TOKEN}" \
    -H "content-type: application/json" \
    --data "{\"repositoryRid\":\"${REPO_RID}\"}")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "HTTP $HTTP_CODE"

if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: Git token request failed"
    echo "$BODY" | jq .
    return 1 2>/dev/null || exit 1
fi

FOUNDRY_GIT_TOKEN=$(echo "$BODY" | jq -r '.token // .gitToken // .')
FOUNDRY_GIT_USER=$(echo "$BODY" | jq -r '.user // .username // "user"')

export FOUNDRY_GIT_TOKEN
export FOUNDRY_GIT_USER

echo "FOUNDRY_GIT_TOKEN exported (${#FOUNDRY_GIT_TOKEN} chars)"
echo ""
echo "Clone with:"
echo "  git clone https://${FOUNDRY_GIT_USER}:\${FOUNDRY_GIT_TOKEN}@${FOUNDRY_URL#https://}/stemma/git/${REPO_RID}/qa-cell-edge-agent"
