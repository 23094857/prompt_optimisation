from concurrent.futures import ThreadPoolExecutor
import random
from utils.utils import evaluate_single_example

def eval_prompt(prompt, llm_evaluation, dataset, max_evals=None, logger=None) -> float:
    if len(dataset) == 0:
        if logger:
            logger.error("Dataset is empty. Cannot evaluate.")
        return 0.0
    num_samples = int(min(max_evals, len(dataset)) if max_evals is not None else len(dataset))
    num_samples = max(num_samples, 1)
    test_indices = random.sample(range(len(dataset)), num_samples)

    def evaluate_single(idx):
        data = dataset[idx]
        return evaluate_single_example(data['question'], data['answer'], prompt, llm_evaluation, logger)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(evaluate_single, test_indices))

    accuracy = sum(results) / num_samples

    if logger:
        logger.info(f"Finished evaluation. Prompt: '{prompt}', Dataset: {dataset.dataset_name}, Mode: {dataset.dataset_mode}, Samples: {num_samples}, Accuracy: {accuracy:.4f}")

    return accuracy
