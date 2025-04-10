import os
from dotenv import load_dotenv
from database import (
    get_db_connection, insert_document, insert_clauses, 
    insert_embeddings, find_similar_clauses
)
from embedding_processor_alternative import generate_embedding
from app import detect_anomalies_simple

# Load environment variables
load_dotenv()

def test_complete_workflow():
    """Test the entire workflow from API calls to database operations"""
    print("=" * 50)
    print("TESTING COMPLETE LEGAL ANOMALY DETECTION WORKFLOW")
    print("=" * 50)
    
    # Step 1: Test database connection
    print("\n📊 STEP 1: Testing database connection...")
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to database")
        return False
    
    print("✅ Successfully connected to database")
    conn.close()
    
    # Step 2: Process a sample contract with mock OpenAI
    print("\n🤖 STEP 2: Processing sample contract with mock OpenAI API...")
    sample_contract = """
    TERMS OF SERVICE

    1. Service Usage
    The Company reserves the right to terminate service at any time for any reason without notice.

    2. Payment
    Users agree to pay all fees, even those resulting from billing errors or system malfunctions.

    3. Liability
    Under no circumstances shall the Company be liable for any damages whatsoever, even if previously advised of such possibility.

    4. Dispute Resolution
    Any disputes shall be resolved exclusively through arbitration in a location of the Company's choosing, regardless of convenience to the user.
    """
    
    # Get analysis from mock OpenAI
    analysis_result = detect_anomalies_simple(sample_contract)
    print("\nAnalysis Result:")
    print("-" * 40)
    print(analysis_result)
    print("-" * 40)
    
    # Step 3: Store document in database
    print("\n💾 STEP 3: Storing document in database...")
    doc_id = insert_document(
        "Sample Terms of Service", 
        "Terms of Service", 
        sample_contract
    )
    
    if not doc_id:
        print("❌ Failed to insert document")
        return False
    
    print(f"✅ Document inserted with ID: {doc_id}")
    
    # Step 4: Extract and store clauses
    print("\n📝 STEP 4: Extracting and storing clauses...")
    
    # In a real application, you might extract these from the analysis result
    # For this test, we'll manually define them
    clauses = [
        (doc_id, "The Company reserves the right to terminate service at any time for any reason without notice.", 
         "Termination", True, "Potentially unfair - no notice required"),
        
        (doc_id, "Users agree to pay all fees, even those resulting from billing errors or system malfunctions.", 
         "Payment", True, "Unfair - holds users responsible for system errors"),
        
        (doc_id, "Under no circumstances shall the Company be liable for any damages whatsoever, even if previously advised of such possibility.", 
         "Liability", True, "Overly broad liability limitation"),
        
        (doc_id, "Any disputes shall be resolved exclusively through arbitration in a location of the Company's choosing, regardless of convenience to the user.", 
         "Dispute Resolution", True, "Unfair arbitration clause")
    ]
    
    clause_ids = insert_clauses(clauses)
    if not clause_ids:
        print("❌ Failed to insert clauses")
        return False
    
    print(f"✅ Inserted {len(clause_ids)} clauses with IDs: {clause_ids}")
    
    # Step 5: Generate and store embeddings
    print("\n🧠 STEP 5: Generating and storing embeddings...")
    embeddings_data = []
    for i, clause_id in enumerate(clause_ids):
        embedding = generate_embedding(clauses[i][1])
        embeddings_data.append((clause_id, embedding))
    
    if not insert_embeddings(embeddings_data):
        print("❌ Failed to insert embeddings")
        return False
    
    print("✅ Successfully inserted embeddings")
    
    # Step 6: Test similarity search
    print("\n🔍 STEP 6: Testing similarity search...")
    test_queries = [
        "The company is not responsible for damages",
        "Cancellation of service without notice",
        "Payment for errors in the system"
    ]
    
    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        test_embedding = generate_embedding(query)
        similar_clauses = find_similar_clauses(test_embedding, limit=2)
        
        if not similar_clauses:
            print("  ❌ No similar clauses found")
            continue
        
        print("  ✅ Found similar clauses:")
        for clause_id, text, similarity in similar_clauses:
            print(f"    - Similarity: {similarity:.4f}")
            print(f"      Text: {text}")
    
    print("\n" + "=" * 50)
    print("✅ COMPLETE WORKFLOW TEST FINISHED SUCCESSFULLY!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    test_complete_workflow() 