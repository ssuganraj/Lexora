# evaluation/retrieval_eval.py

def recall_at_k(retrieved_pages, ground_truth_page, k):
    return int(ground_truth_page in retrieved_pages[:k])


def evaluate_retriever(retriever, eval_data, k=3):
    hits = 0

    for sample in eval_data:
        results = retriever.retrieve(sample["question"], top_k=k)
        pages = [page for page, _, _ in results]

        hits += recall_at_k(
            pages,
            sample["answer_page"],
            k
        )

    return hits / len(eval_data)
