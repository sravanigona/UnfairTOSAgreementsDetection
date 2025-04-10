import os
from dotenv import load_dotenv
from database import get_db_connection, insert_document, insert_clauses, insert_embeddings, find_similar_clauses
from embedding_processor_alternative import generate_embedding

# Load environment variables
load_dotenv()

def test_full_workflow():
    """Test the entire workflow from document insertion to similarity search"""
    print("Testing connection to Docker PostgreSQL database...")
    
    # Test database connection
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to database")
        return False
    
    print("✅ Successfully connected to database")
    conn.close()
    
    # Insert a test document
    print("\nInserting test document...")
    doc_id = insert_document(
        "Sample Contract", 
        "Terms of Service", 
        "This is a sample contract with various clauses for testing."
    )
    
    if not doc_id:
        print("❌ Failed to insert document")
        return False
    
    print(f"✅ Document inserted with ID: {doc_id}")
    
    # Insert test clauses
    print("\nInserting test clauses...")
    clauses = [
        (doc_id, "The Company reserves the right to terminate service at any time.", "Termination", False, None),
        (doc_id, "Users must pay all fees within 30 days of invoice.", "Payment", False, None),
        (doc_id, "Users agree to pay all fees, even those resulting from billing errors.", "Payment", True, "Potentially unfair term"),
        (doc_id, "Any disputes shall be resolved through arbitration.", "Dispute", False, None),
        (doc_id, "The Company is not liable for any damages whatsoever.", "Liability", True, "Overly broad limitation")
    ]
    
    clause_ids = insert_clauses(clauses)
    if not clause_ids:
        print("❌ Failed to insert clauses")
        return False
    
    print(f"✅ Inserted {len(clause_ids)} clauses with IDs: {clause_ids}")
    
    # Generate and insert embeddings
    print("\nGenerating and inserting embeddings...")
    embeddings_data = []
    for i, clause_id in enumerate(clause_ids):
        # Generate embedding for clause text
        embedding = generate_embedding(clauses[i][1])  
        embeddings_data.append((clause_id, embedding))
    
    if not insert_embeddings(embeddings_data):
        print("❌ Failed to insert embeddings")
        return False
    
    print("✅ Successfully inserted embeddings")
    
    # Test similarity search
    print("\nTesting similarity search...")
    test_text = "The company will not be responsible for any damages"
    test_embedding = generate_embedding(test_text)
    
    similar_clauses = find_similar_clauses(test_embedding, limit=3)
    if not similar_clauses:
        print("❌ No similar clauses found")
        return False
    
    print("✅ Found similar clauses:")
    for clause_id, text, similarity in similar_clauses:
        print(f"  - ID: {clause_id}, Similarity: {similarity:.4f}")
        print(f"    Text: {text}")
    
    print("\n✅ Full workflow test completed successfully!")
    return True

if __name__ == "__main__":
    test_full_workflow() 