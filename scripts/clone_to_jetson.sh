#!/bin/bash
# Clone the repo onto the Jetson Nano using a personal Foundry token.
#
# Run this ON THE JETSON (or any fresh machine). It will:
#   1. Exchange your personal token for a git token
#   2. Clone the repo to /opt/qa-cell-edge-agent
#   3. Set up the venv and install dependencies
#
# Usage:
#   curl -s <this-script-url> | bash -s -- eyJ...  # one-liner
#   bash clone_to_jetson.sh eyJ...                  # or download first
#
# Requires: curl, jq, git, python3

set -e

FOUNDRY_URL="${FOUNDRY_URL:-https://regalia.palantircloud.com}"
REPO_RID="${REPO_RID:-ri.stemma.main.repository.ddb28091-90ea-49fb-b207-c9e55ecd1d8e}"
INSTALL_DIR="/opt/qa-cell-edge-agent"

# ── Get personal token ───────────────────────────────────────────
PALANTIR_TOKEN="${1:-}"
if [ -z "$PALANTIR_TOKEN" ]; then
    echo -n "Paste your Foundry personal token: "
    read -s PALANTIR_TOKEN
    echo ""
fi

# ── Exchange for git token ───────────────────────────────────────
echo "Requesting git token..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${FOUNDRY_URL}/code/api/security/gitToken" \
    -H "accept: application/json" \
    -H "authorization: Bearer ${PALANTIR_TOKEN}" \
    -H "content-type: application/json" \
    --data "{\"repositoryRid\":\"${REPO_RID}\"}")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
    echo "ERROR: Git token request failed (HTTP $HTTP_CODE)"
    echo "$BODY" | jq . 2>/dev/null || echo "$BODY"
    exit 1
fi

GIT_TOKEN=$(echo "$BODY" | jq -r '.token // .gitToken // .')
echo "Git token OK"

# ── Clone ────────────────────────────────────────────────────────
echo "Cloning to ${INSTALL_DIR}..."
sudo mkdir -p "$(dirname "$INSTALL_DIR")"
sudo git clone "https://user:${GIT_TOKEN}@${FOUNDRY_URL#https://}/stemma/git/${REPO_RID}/qa-cell-edge-agent" "$INSTALL_DIR"
sudo chown -R "$(whoami):$(id -gn)" "$INSTALL_DIR"

# ── Setup ────────────────────────────────────────────────────────
cd "$INSTALL_DIR"
echo "Creating venv..."
python3 -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -e src/

# ── Serial port permissions ──────────────────────────────────────
if ! groups | grep -q dialout; then
    echo "Adding $(whoami) to dialout group (serial port access)..."
    sudo usermod -aG dialout "$(whoami)"
    echo "NOTE: Log out and back in for group change to take effect."
fi

# ── Done ─────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Clone complete: ${INSTALL_DIR}"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "    cd ${INSTALL_DIR}"
echo "    source venv/bin/activate"
echo "    cp .env.example .env              # fill in Foundry credentials"
echo "    python scripts/register_robot.py  # register robot in Foundry"
echo "    python scripts/verify_hardware.py # check serial + camera"
echo "    python scripts/calibrate_arm.py   # calibrate waypoints"
echo "    python scripts/test_connection.py --seed --count 10"
echo "    python -m qa_cell_edge_agent.main"
echo ""
