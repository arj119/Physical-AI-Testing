#!/bin/bash
# Fetch a Foundry OAuth2 token and export as FOUNDRY_TOKEN.
# Usage: source scripts/get_token.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env (only the vars we need)
if [ -f "$PROJECT_DIR/.env" ]; then
    FOUNDRY_URL=$(grep '^FOUNDRY_URL=' "$PROJECT_DIR/.env" | cut -d= -f2-)
    CLIENT_ID=$(grep '^CLIENT_ID=' "$PROJECT_DIR/.env" | cut -d= -f2-)
    CLIENT_SECRET=$(grep '^CLIENT_SECRET=' "$PROJECT_DIR/.env" | cut -d= -f2-)
fi

echo "Requesting token from ${FOUNDRY_URL}..."

RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${FOUNDRY_URL}/multipass/api/oauth2/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "grant_type=client_credentials&client_id=${CLIENT_ID}&client_secret=${CLIENT_SECRET}&scope=api:use-streams-write")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "HTTP $HTTP_CODE"

if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: Token request failed"
    echo "$BODY" | jq .
    return 1 2>/dev/null || exit 1
fi

FOUNDRY_TOKEN=$(echo "$BODY" | jq -r '.access_token')
EXPIRES_IN=$(echo "$BODY" | jq -r '.expires_in')

export FOUNDRY_TOKEN
echo "FOUNDRY_TOKEN exported (${#FOUNDRY_TOKEN} chars, expires in ${EXPIRES_IN}s)"
