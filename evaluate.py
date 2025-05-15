# evaluate.py
from rag_utils import create_rag_chain
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from difflib import SequenceMatcher
import numpy as np

def load_eval_data(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def similarity_score(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def evaluate_rag_system(eval_data, rag_chain, threshold=0.8):
    y_true, y_pred, similarity_scores = [], [], []

    for entry in eval_data:
        query, expected = entry["query"], entry["expected_answer"]
        response = rag_chain.run(query)
        sim_score = similarity_score(expected, response)
        similarity_scores.append(sim_score)

        match = 1 if sim_score >= threshold else 0
        y_true.append(1)
        y_pred.append(match)

        print(f"\nQ: {query}")
        print(f"Expected: {expected}")
        print(f"Got: {response}")
        print(f"Sim: {sim_score:.2f} => {'✔' if match else '✘'}")

    print("\n--- Metrics ---")
    print(f"Precision: {precision_score(y_true, y_pred):.2f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.2f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.2f}")
    print(f"Avg Sim:   {np.mean(similarity_scores):.2f}")

def main():
    rag_chain = create_rag_chain()
    eval_data = load_eval_data("evaluation_data.json")
    evaluate_rag_system(eval_data, rag_chain)

if __name__ == "__main__":
    main()
