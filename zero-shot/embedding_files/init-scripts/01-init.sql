-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Drop existing tables if they exist
-- DROP TABLE IF EXISTS clause_embeddings CASCADE;

-- Create the new table
CREATE TABLE IF NOT EXISTS clause_embeddings (
    id SERIAL PRIMARY KEY,
    clause TEXT NOT NULL,
    label TEXT,
    type TEXT,
    embedding_vector vector(1536) NOT NULL,
    train_test_split TEXT NOT NULL, -- Indicates whether it's train, test, or validation
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for similarity search
CREATE INDEX ON clause_embeddings USING ivfflat (embedding_vector vector_cosine_ops);