def dict_has_at_least_two_nonzero_values(probabilities: dict) -> bool:
    """Return True if there are at least two non-zero entries in the dictionary."""
    non_zero_count = sum(1 for value in probabilities.values() if value > 0)
    return non_zero_count >= 2

def evaluate_single_example(question, correct_answer, prompt, llm_evaluation, logger=None):
    predicted_answer = llm_evaluation.predict(f"{question} {prompt}", logger=logger)

    expected_pattern = "=" + correct_answer.strip()
    correct = expected_pattern in "".join(predicted_answer.split())

    if logger:
        logger.info(f"Evaluating: {question}")
        logger.info(f"Expected: {correct_answer.strip()}")
        logger.info(f"Predicted: {predicted_answer.strip()}")
        logger.info(f"Correct: {correct}")

    return correct
