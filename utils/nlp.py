import random
from functools import lru_cache
import nltk
        
nltk.download('brown', quiet=True)

@lru_cache(maxsize=None)
def get_cumulative_dist(exclude_top_n=0, print_top_n=0):
    brown_words = nltk.corpus.brown.words()
    freq_dist = nltk.probability.FreqDist(brown_words)
    total_count = sum(freq_dist.values())
    prob_dist = {word: freq / total_count for word, freq in freq_dist.items()}

    if print_top_n > 0:
        print(f"{print_top_n} most common words in the Brown corpus:")
        for word, freq in freq_dist.most_common(print_top_n):
            print(f"{word}: {freq / total_count:.4f}")

    # Exclude the top N most frequent words if exclude_top_n > 0
    if exclude_top_n > 0:
        words_to_exclude = set(word for word, _ in freq_dist.most_common(exclude_top_n))
        prob_dist = {word: freq for word, freq in prob_dist.items() if word not in words_to_exclude}
        # Recalculate total count after exclusion
        total_count = sum(prob_dist.values())
        prob_dist = {word: freq / total_count for word, freq in prob_dist.items()}

    cumulative_dist = []
    cumulative = 0
    for word, prob in sorted(prob_dist.items(), key=lambda x: -x[1]):  # Sort by probability for stable cumulative order
        cumulative += prob
        cumulative_dist.append((cumulative, word))

    return cumulative_dist
    

def sample_word(exclude_top_n=20):
    cumulative_dist = get_cumulative_dist(exclude_top_n=exclude_top_n)
    
    rand = random.random()
    for cum_prob, word in cumulative_dist:
        if rand < cum_prob:
            return word
        
def make_valid_instruction(sentence, llm_assistent, logger):
    user_prompt = ("Check if the given sentence is a grammatically correct instruction. "
               "If not, make the smallest change to correct it. "
               "The new sentence should be 2 to 10 words long. "
               "Make sure the output is only the new sentence. "
               "Input sentence: " + sentence +
               "Output sentence: ")
    answer = llm_assistent.predict(user_prompt, logger)
    #remove spaces and linebreaks in beginning and end from answer
    cleaned_answer = answer.strip()

    logger.info(f'make_valid_instruction(), IN,{sentence},OUT,{cleaned_answer}')

    return cleaned_answer