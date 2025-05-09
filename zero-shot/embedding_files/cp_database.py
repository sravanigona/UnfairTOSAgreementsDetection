import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


load_dotenv()

DB_NAME = os.getenv("DB_NAME", "legal_anomaly_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")


def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None


def setup_database():
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cursor:
            # The tables are already created by the init script in Docker
            # This is just a verification step now
            cursor.execute("SELECT to_regclass('legal_documents');")
            has_tables = cursor.fetchone()[0] is not None

            if has_tables:
                print("Database tables already exist.")
                return True

            print("Tables don't exist. This shouldn't happen with the Docker setup.")
            return False
    except Exception as e:
        print(f"Database verification error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def insert_document(title, document_type, content):
    conn = get_db_connection()
    if not conn:
        return None

    try:
        with conn.cursor() as cursor:
            # Insert legal document
            cursor.execute("""
                INSERT INTO legal_documents (title, document_type, content)
                VALUES (%s, %s, %s)
                RETURNING id;
            """, (title, document_type, content))

            document_id = cursor.fetchone()[0]
            conn.commit()  # commment this out if you are testing and not trying to
            print(f"Legal document inserted with ID: {document_id}")

            return document_id
    except Exception as e:
        print(f"Error inserting document: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()


def insert_clauses(clauses_data):
    """
    Insert multiple clauses into the database.
    clauses_data should be a list of tuples: (document_id, clause_text, clause_type, is_anomalous, anomaly_description)
    """
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cursor:
            execute_values(
                cursor,
                "INSERT INTO document_clauses (document_id, clause_text, clause_type, is_anomalous, anomaly_description) VALUES %s RETURNING id;",
                clauses_data
            )
            clause_ids = [row[0] for row in cursor.fetchall()]
            conn.commit()
            return clause_ids
    except Exception as e:
        print(f"Error inserting clauses: {e}")
        conn.rollback()
        return []
    finally:
        conn.close()


def insert_embeddings(embeddings_data):
    """
    Insert embeddings into the database.
    embeddings_data should be a list of tuples: (clause_id, embedding_vector)
    """
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cursor:
            # Convert each embedding to a PostgreSQL vector
            for i, (clause_id, embedding) in enumerate(embeddings_data):
                cursor.execute(
                    "INSERT INTO embeddings (clause_id, embedding_vector) VALUES (%s, %s);",
                    (clause_id, embedding)
                )
            conn.commit()
            return True
    except Exception as e:
        print(f"Error inserting embeddings: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

# Add a function for similarity search


def find_similar_clauses(embedding, limit=5):
    """
    Find similar clauses using cosine similarity.
    """
    conn = get_db_connection()
    if not conn:
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT c.id, c.clause_text, 1 - (e.embedding_vector <=> %s) as similarity
                FROM embeddings e
                JOIN document_clauses c ON e.clause_id = c.id
                ORDER BY e.embedding_vector <=> %s
                LIMIT %s;
            """, (embedding, embedding, limit))

            results = cursor.fetchall()
            return [(id, text, similarity) for id, text, similarity in results]
    except Exception as e:
        print(f"Error in similarity search: {e}")
        return []
    finally:
        conn.close()


# Initialize database if this script is run directly
if __name__ == "__main__":
    setup_database()
