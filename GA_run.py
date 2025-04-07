#! /usr/bin/env python3

from GA import GA
from utils.llm import LLM
from utils.data_loader import DataLoader


llm_evaluation = LLM("fireworks-llama-v3p1-8b-instruct", request_timeout=50000, temperature=1)
llm_assistent = LLM("fireworks-llama-v3-70b-instruct", request_timeout=50000, temperature=1)
dataset = DataLoader(dataset_name="math", mode="train")

population_HPS = [
    'The governor of the university needs able citizens above all.',
    'Study science for others at Saracens Presidency.',
    'Review both.',
    'Read by Gyp is corrected to: Read it by Gyp.',
    'Come to the last spot to mount the situation.',
    'Robert saw a capillary with a head.',
    'Choose on Monday.',
    'The statement says on the output.'
 ]

population_HPS_2 = [
    "Parents should promote.",
    "Listen to this teacher!",
    "Do not think about the world of work.",
    "Walk on this path this year.",
    "Only then is the time for preparation and knowledge.",
    "Go see Mr. address the government in town.",
    "Rent these findings to those having difficulty lines.",
    "He excels in industrial which."
]


population_best5_GA = [
    "Study science for your test in Saracens' machinery class.",
    "Review it in Italy.",
    "Read for the L'Turu Saracens Presidency.",
    "Review the time in others' work on Mondays.",
    "Do it on this day."
]

# GA_object = GA(llm_evaluation, llm_assistent, 'math', seed=42, split=0.9,
#                 population = population_HPS_2,
#                 # max_generations=0,
#                 evaluations_per_prompt=250,
#                 max_evaluations_per_prompt=250,
#                 population_size=8,
#                 baseline_accuracy=0.4512,
#                 n_final_evals_per_prompt=1000,
#                 max_llm_calls=int(50000),
#                 mutation_probability=0.2937385571865594,
#                 cross_over_ratio=0.31250447802472814,
#                 # reset_evals_every_n_llm_calls=40000,
#                 n_testset_evals=1000
#             )

GA_object = GA(llm_evaluation, llm_assistent, 'math', seed=42, split=0.9,
                population = population_HPS_2,
                # max_generations=0,
                evaluations_per_prompt=5,
                max_evaluations_per_prompt=5,
                population_size=4,
                baseline_accuracy=0.4512,
                n_final_evals_per_prompt=5,
                max_llm_calls=int(100),
                mutation_probability=0.2937385571865594,
                cross_over_ratio=0.31250447802472814,
                # reset_evals_every_n_llm_calls=40000,
                n_testset_evals=5
            )

GA_object.run_GA()




