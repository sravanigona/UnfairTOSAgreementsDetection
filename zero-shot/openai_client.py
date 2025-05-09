import os
from dotenv import load_dotenv
import openai
import numpy as np
from pprint import pprint
from database import insert_clause_embeddings
from dataset_functions import load_claudette_tos_dataset, estimate_tokens

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai


def test_single_clause_analysis(contract_text):
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


def test_openAI_connection(clauses):
    print(openai.api_key, os.getenv("OPENAI_API_KEY"))
    print("----" * 50)
    """
    Simple test to check that there is a connection with the openAI API.
    """
    system_prompt = """ You are a basketball coach who has been asked a question
    and you will give a motivational speech to inspire
    the 7th graders who are currently losing at half time.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Classify Clauses: {clauses}"}
            ],
            temperature=0.0,
            max_tokens=150
        )
        print("=" * 50)
        pprint(response)
        print("=" * 50)
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"


def test_embedding_insert():

    splits = load_claudette_tos_dataset()
    clauses = splits['train']['text'][:5]  # Use a small subset for testing
    labels = splits['train']['label'][:5]  # Corresponding labels

    train_test_type = "train"
    # Use mock embeddings for testing
    embeddings = []
    # Prepare data for insertion
    data = [
        # Replace `None` with `type` if applicable
        (clause, label, None, embedding, train_test_type)
        for clause, label, embedding in zip(clauses, labels, embeddings)
    ]

    # Insert data into the database
    success = insert_clause_embeddings(data)
    if success:
        print("Embeddings inserted successfully into the database.")
    else:
        print("Failed to insert embeddings into the database.")


def classify_clauses_batch(clauses, system_prompt, batch_size=100, model="gpt-4o-mini", mode="test"):
    shouldBreak = False
    if mode == "test":
        batch_size = 10
        shouldBreak = True
    all_predictions = []
    for i in range(0, len(clauses), batch_size):
        batch_clauses = clauses[i: i + batch_size]
        user_prompt = "Classify the following clauses:\n" + "\n".join(
            [f"{j+1}. {clause}" for j, clause in enumerate(batch_clauses)]
        )
        full_prompt = system_prompt + "\n" + user_prompt
        token_count = estimate_tokens(full_prompt, model=model)
        print(
            f"Estimated token count for batch {i // batch_size + 1}: {token_count}")

        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0,  # For consistent results
            )
            # Extract classifications from the response
            batch_predictions = []
            print(response)
            for choice in response.choices:
                message_content = choice.message.content
                # print(message_content)
                batch_predictions.append(message_content)
                # print(batch_predictions)
            all_predictions.extend(batch_predictions)
            print(f"Processed batch {i // batch_size + 1}")
            if shouldBreak:
                break
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
    return all_predictions


def collect_labels(all_predictions):
    print(len(all_predictions))
    output_labels = []
    for message in all_predictions:
        lst = message.split("\n")
        for predict in lst:
            output_labels.append(
                int(predict.strip().split('Classification: ')[1]))
    return output_labels


def main():
    system_prompt = """
    You are a legal expert in consumer rights and contract law. Your task is to classify clauses from Terms of Service (ToS) documents as either '0' (Fair) or '1' (Unfair).

    A clause is '1' (Unfair) if it:
    1. Imposes unreasonable restrictions or obligations on users.
    2. Significantly limits users' rights or remedies.
    3. Creates an imbalance of power favoring the service provider.
    4. Uses vague or ambiguous language that could be exploited.
    5. Violates established legal principles or consumer protection laws.

    If none of these apply, classify the clause as '0' (Fair). For each clause, You do not have to number it. You can provide the output line by line in the
    following format so it is easy to parse:

    Classification: <0 or 1>
    """

    final_splits = load_claudette_tos_dataset()

    validation_clauses = final_splits['validation']['text']
    validation_labels = final_splits['validation']['label']

    train_clauses = final_splits['train']['text']
    train_labels = final_splits['validation']['label']

    test_clauses = final_splits['test']['text']
    test_labels = final_splits['validation']['label']

    classify_clauses_batch()

    return


if __name__ == "__main__":
    print("=" * 50)
    # result = test_openAI_connection()
    # print(result)
    # embed_clauses()
    # test_embedding_insert()
