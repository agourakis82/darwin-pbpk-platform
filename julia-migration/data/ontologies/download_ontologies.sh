#!/bin/bash
# Download OBO Foundry ontologies for Darwin PBPK Platform
# These are large files - not tracked in git

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Downloading OBO Foundry ontologies..."

# DOID - Disease Ontology (required for disease-drug associations)
echo "Downloading DOID (Disease Ontology)..."
curl -L -o doid.owl "http://purl.obolibrary.org/obo/doid.owl"
curl -L -o doid.json "http://purl.obolibrary.org/obo/doid.json"
curl -L -o doid.obo "http://purl.obolibrary.org/obo/doid.obo"

# ChEBI - Chemical Entities of Biological Interest (optional - very large ~500MB)
# Uncomment if needed:
# echo "Downloading ChEBI..."
# curl -L -o chebi.owl "http://purl.obolibrary.org/obo/chebi.owl"

# DINTO - Drug-Drug Interaction Ontology (optional)
# echo "Downloading DINTO..."
# curl -L -o dinto.owl "http://purl.obolibrary.org/obo/dinto.owl"

# UO - Units Ontology
echo "Downloading UO (Units Ontology)..."
curl -L -o uo.owl "http://purl.obolibrary.org/obo/uo.owl"

# OBI - Ontology for Biomedical Investigations
echo "Downloading OBI..."
curl -L -o obi.owl "http://purl.obolibrary.org/obo/obi.owl"

echo ""
echo "Download complete. Files:"
ls -lh *.owl *.json *.obo 2>/dev/null || true

echo ""
echo "DOID version:"
head -10 doid.obo | grep -E "^(data-version|date)" || true
