-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables
CREATE TABLE IF NOT EXISTS legal_documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    document_type TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_clauses (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES legal_documents(id),
    clause_text TEXT NOT NULL,
    clause_type TEXT,
    is_anomalous BOOLEAN,
    anomaly_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Use vector type for embeddings
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    clause_id INTEGER REFERENCES document_clauses(id),
    embedding_vector vector(384) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for similarity search
CREATE INDEX ON embeddings USING ivfflat (embedding_vector vector_cosine_ops); 