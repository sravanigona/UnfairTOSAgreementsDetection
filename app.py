import os 
from dotenv import load_dotenv


load_dotenv()

# Check if we should use mock or real OpenAI
USE_MOCK = os.getenv("USE_MOCK", "true").lower() == "true"

if USE_MOCK:
    from mock_openai import mock_openai as client
else:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY", "dummy-api-key")
    client = openai

def detect_anomalies_simple(contract_text):

    """
    Simple benchmark using OpenAI API to detect legal anomalies in contract text.
    """
    system_prompt = """
    You are a legal expert specialized in contract analysis. Your task is to identify anomalies 
    in legal documents, contracts, or terms of service. Anomalies include:
    
    1. Unusual clauses that deviate from standard practice
    2. Potentially unfair terms
    3. Vague or ambiguous language that could be exploited
    4. Contradictions within the document
    5. Terms that may not be legally enforceable
    
    Analyze the provided text and classify whether it contains anomalies. If anomalies are found,
    specify their nature and potential implications.
    """

    try: 
        if USE_MOCK:
            response = client.chat_completion_create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze the following contract text for anomalies:\n\n{contract_text}"}
                ],
                temperature=0.0
            )
        else:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze the following contract text for anomalies:\n\n{contract_text}"}
                ],
                temperature=0.0
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
    
# Example usage with dummy data
if __name__ == "__main__":
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

    result = detect_anomalies_simple(sample_contract)
    print(result)



