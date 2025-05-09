import os
from dotenv import load_dotenv
import openai
from pprint import pprint
from dataset_functions import load_claudette_tos_dataset, estimate_tokens
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai


def classify_clauses_batch(clauses, system_prompt, batch_size=10, model="gpt-4o-mini", mode="test"):
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
            for choice in response.choices:
                message_content = choice.message.content
                batch_predictions.extend(message_content.strip().split("\n"))

            # Verify the number of classifications matches the batch size
            if len(batch_predictions) != len(batch_clauses):
                print(
                    f"WARNING: Mismatch in batch size for Batch {i // batch_size + 1}!")
                print(
                    f"Expected {len(batch_clauses)} classifications, but got {len(batch_predictions)}.")

            # Append verified predictions to the final list
            all_predictions.extend(batch_predictions)
            print(
                f"Batch {i // batch_size + 1} processed successfully with {len(batch_predictions)} classifications.")

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
    return all_predictions


def evaluate_predictions(ground_truth, predictions):
    """
    Evaluates the predictions against ground truth labels and calculates metrics.

    Args:
      ground_truth: A list of ground truth labels (0 for Fair, 1 for Unfair).
      predictions: A list of predicted labels (0 for Fair, 1 for Unfair).

    Returns:
      A dictionary containing the calculated metrics.
    """

    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)

    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    return metrics


def main():
    # system_prompt = """
    # You are a legal expert in consumer rights and contract law. Your task is to classify clauses from Terms of Service (ToS) documents as either '0' (Fair) or '1' (Unfair).

    # A clause is '1' (Unfair) if it:
    # 1. Imposes unreasonable restrictions or obligations on users.
    # 2. Significantly limits users' rights or remedies.
    # 3. Creates an imbalance of power favoring the service provider.
    # 4. Uses vague or ambiguous language that could be exploited.
    # 5. Violates established legal principles or consumer protection laws.

    # If none of these apply, classify the clause as '0' (Fair). For each clause, You do not have to number it. You can provide the output line by line in the
    # following format so it is easy to parse:

    # Classification: <0 or 1>
    # """

    system_prompt = """
    You are a legal expert in consumer rights and contract law. Your task is to classify clauses from Terms of Service (ToS) 
    documents as either '0' (Fair) or '1' (Unfair).

    A clause is '1' (Unfair) if it:
    1. Imposes unreasonable restrictions or obligations on users.
    2. Significantly limits users' rights or remedies.
    3. Creates an imbalance of power favoring the service provider.
    4. Uses vague or ambiguous language that could be exploited.
    5. Violates established legal principles or consumer protection laws.

    Classify ONLY the clauses provided in the user prompt. Provide ONLY the classification ('Classification: 0' or 'Classification: 1') 
    for each clause, line by line, in the same order as the input. Do not add extra lines, blank lines, or explanations.
    """

    final_splits = load_claudette_tos_dataset()

    validation_clauses = final_splits['validation']['text']
    validation_labels = final_splits['validation']['label']

    train_clauses = final_splits['train']['text']
    train_labels = final_splits['validation']['label']

    test_clauses = final_splits['test']['text']
    test_labels = final_splits['validation']['label']

    validation_predictions = classify_clauses_batch(
        validation_clauses, system_prompt)

    # evaluate_predictions(validation_labels, validation_predicted_labels)


if __name__ == '__main__':
    # main()
    print()
