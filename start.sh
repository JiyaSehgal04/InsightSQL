#!/bin/bash
set -e

# Seed Northwind schema on first boot (checks if the orders table exists)
TABLE_EXISTS=$(psql "$DATABASE_URL" -tAc "SELECT to_regclass('public.orders')" 2>/dev/null || true)
if [ -z "$TABLE_EXISTS" ]; then
    echo "Seeding Northwind database..."
    psql "$DATABASE_URL" -f data/postgres/init/01_northwind_demo.sql
    echo "Database seeded."
fi

# Build ChromaDB RAG index on first boot (rebuilds from static Python data, ~10s)
if [ ! -d ".rag_index" ]; then
    echo "Building RAG index..."
    python scripts/build_rag_index.py
    echo "RAG index ready."
fi

exec streamlit run frontend/streamlit_app.py \
    --server.port="${PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true
