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


def test_database():
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cursor:
            # Check if the `clause_embeddings` table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'clause_embeddings'
                );
            """)
            table_exists = cursor.fetchone()[0]

            if table_exists:
                print("The `clause_embeddings` table exists.")
                return True

            print("The `clause_embeddings` table does not exist.")
            return False
    except Exception as e:
        print(f"Database verification error: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def insert_clause_embeddings(data):
    """
    Insert multiple clause embeddings into the `clause_embeddings` table.
    :param data: List of tuples (clause, label, type, embedding_vector, train_test_split)
    """
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cursor:
            query = """
                INSERT INTO clause_embeddings (clause, label, type, embedding_vector, train_test_split)
                VALUES %s
            """
            execute_values(cursor, query, data)
        conn.commit()
        print("Data inserted successfully.")
        return True
    except Exception as e:
        print(f"Error inserting data: {e}")
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
    test_database()
