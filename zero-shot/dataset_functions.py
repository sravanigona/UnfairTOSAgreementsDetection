import os
import openai
from dotenv import load_dotenv
from database import insert_clause_embeddings
from datasets import load_dataset, DatasetDict
import tiktoken


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai


def estimate_tokens(prompt, model="gpt-4o-mini"):
    """
    Estimate the number of tokens in a given prompt for a specific model.
    :param prompt: The text prompt to estimate tokens for.
    :param model: The model name (e.g., "gpt-4", "gpt-3.5-turbo").
    :return: The estimated token count.
    """
    # Load the tokenizer for the specified model
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(prompt)
    return len(tokens)


def load_claudette_tos_dataset():
    dataset = load_dataset("LawInformedAI/claudette_tos")['train']
    dataset = dataset.class_encode_column('label')
    split1 = dataset.train_test_split(
        test_size=0.2,
        stratify_by_column='label',
        seed=42
    )
    split2 = split1['test'].train_test_split(
        test_size=0.5,
        stratify_by_column='label',
        seed=42
    )
    # Combine splits
    final_splits = DatasetDict({
        'train': split1['train'],
        'validation': split2['train'],
        'test': split2['test']
    })

    return final_splits


def embed_clauses_in_batches(batch_size=100):
    """
    Embed clauses in batches and insert them into the database.
    :param batch_size: Number of clauses to process per batch.
    """

    # only embedding the training data for now
    # Load the dataset
    splits = load_claudette_tos_dataset()
    clauses = splits['train']['text']
    labels = splits['train']['label']
    train_test_type = "train"

    total_clauses = len(clauses)
    print(f"Total clauses to embed: {total_clauses}")

    isFirst = True

    for i in range(0, total_clauses, batch_size):
        # Get the current batch
        batch_clauses = clauses[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        # Generate embeddings for the batch
        try:
            embeddings_response = generate_embeddings(batch_clauses)
            embeddings = [item.embedding
                          for item in embeddings_response.data]

            # Prepare data for insertion
            data = [
                (clause, label, None, embedding, train_test_type)
                for clause, label, embedding in zip(batch_clauses, batch_labels, embeddings)
            ]

            # Insert embeddings into the database
            success = insert_clause_embeddings(data)
            if success:
                print(f"Batch {i // batch_size + 1} inserted successfully.")
            else:
                print(f"Failed to insert batch {i // batch_size + 1}.")
            if isFirst:
                x = input("waiting before continuing....")
                isFirst = False
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")

    print("Embedding process completed.")

# so right now even though it says clauses it is configured for a single clause
# I will need to do it for batches of clauses (that batch number idk)


def generate_embeddings(clauses):
    """
    Generate embeddings for a list of clauses using OpenAI's embedding model.
    :param clauses: List of strings (clauses) to embed.
    :return: Response object containing embeddings or None if an error occurs.
    """
    try:
        # Ensure clauses is a list of strings
        if not isinstance(clauses, list) or not all(isinstance(clause, str) for clause in clauses):
            raise ValueError("Input must be a list of strings.")

        # Call OpenAI's embedding API
        response = client.embeddings.create(
            input=clauses,
            model="text-embedding-3-small"  # Ensure this is a valid model name
        )
        return response
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


if __name__ == "__main__":
    # embed_clauses_in_batches()
    print()
