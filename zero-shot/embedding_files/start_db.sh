#!/bin/bash
echo "Starting PostgreSQL with pgvector..."
docker-compose up -d
echo "Database is now running at localhost:5432" 