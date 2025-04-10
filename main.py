import os
import argparse
from dotenv import load_dotenv
# Import database functions
from database import setup_database, insert_document

# Load environment variables
load_dotenv()

# Check if we should use mock or real OpenAI
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"

if USE_MOCK:
    from mock_openai import MockOpenAI
    client = MockOpenAI()
else:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY", "dummy-api-key")
    client = openai


def analyze_text(text):
    """Simple function to analyze text for legal anomalies"""
    system_prompt = """
    You are a legal expert specialized in contract analysis. Identify anomalies in legal documents
    such as unusual clauses, unfair terms, vague language, contradictions, or unenforceable terms.
    """
    
    try:
        if USE_MOCK:
            response = client.chat_completion_create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text:\n\n{text}"}
                ],
                temperature=0.0
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this text:\n\n{text}"}
                ],
                temperature=0.0
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def test_database():
    """Test database connection and operations"""
    print("Setting up database...")
    if setup_database():
        print("Database setup successful!")
        
        # Test document insertion
        doc_id = insert_document(
            "Test Contract", 
            "Terms of Service", 
            "This is a test contract with some terms and conditions."
        )
        
        if doc_id:
            print(f"Successfully inserted document with ID: {doc_id}")
            return True
        else:
            print("Failed to insert document")
            return False
    else:
        print("Database setup failed")
        return False

def test_database_comprehensive():
    """Test comprehensive database operations with Docker container"""
    from embedding_processor_alternative import generate_embedding
    from database import (
        setup_database, insert_document, insert_clauses, 
        insert_embeddings, find_similar_clauses
    )
    
    print("Testing database operations with Docker container...")
    
    # First verify database connection
    if not setup_database():
        print("Database verification failed")
        return False
    
    # Insert a test document
    doc_id = insert_document(
        "Test Contract", 
        "Terms of Service", 
        "This is a test contract with some terms and conditions."
    )
    
    if not doc_id:
        print("Failed to insert document")
        return False
    
    print(f"Inserted document with ID: {doc_id}")
    
    # Insert test clauses
    clauses = [
        (doc_id, "The user agrees to all terms.", "General", False, None),
        (doc_id, "The company may change terms at any time.", "Amendment", False, None)
    ]
    
    clause_ids = insert_clauses(clauses)
    if not clause_ids or len(clause_ids) != 2:
        print("Failed to insert clauses")
        return False
    
    print(f"Inserted clauses with IDs: {clause_ids}")
    
    # Generate and insert embeddings
    embeddings = []
    for i, clause_id in enumerate(clause_ids):
        embedding = generate_embedding(clauses[i][1])
        embeddings.append((clause_id, embedding))
    
    if not insert_embeddings(embeddings):
        print("Failed to insert embeddings")
        return False
    
    print("Inserted embeddings successfully")
    
    # Test similarity search
    test_embedding = generate_embedding("The company can modify the agreement")
    similar = find_similar_clauses(test_embedding, limit=2)
    
    if not similar:
        print("Failed to find similar clauses")
        return False
    
    print("Found similar clauses:")
    for id, text, similarity in similar:
        print(f"  - {text} (similarity: {similarity:.4f})")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Legal Anomaly Detection')
    parser.add_argument('--test-db', action='store_true', help='Test basic database connection')
    parser.add_argument('--test-full', action='store_true', help='Test full database workflow')
    parser.add_argument('--analyze', type=str, help='Text to analyze for anomalies')
    
    args = parser.parse_args()
    
    if args.test_db:
        test_database()
    
    if args.test_full:
        test_database_comprehensive()
    
    if args.analyze:
        print("\nAnalyzing text:")
        print("-" * 40)
        print(args.analyze)
        print("-" * 40)
        result = analyze_text(args.analyze)
        print("\nAnalysis Result:")
        print(result)

if __name__ == "__main__":
    main() 