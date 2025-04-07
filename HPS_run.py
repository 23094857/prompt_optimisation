#! /usr/bin/env python3

from HPS import HPS

from utils.llm import LLM


llm_evaluation = LLM("fireworks-llama-v3p1-8b-instruct", request_timeout=5000, temperature=1)
llm_assistent = LLM("fireworks-llama-v3-70b-instruct", request_timeout=5000, temperature=1)

# HPS_object = HPS(n_trials=15,
#                     llm_evaluation=llm_evaluation,
#                     llm_assistent=llm_assistent,
#                     dataset=dataset,
#                     population_size=8,
#                     evaluations_per_prompt=250,
#                     max_evaluations_per_prompt=250,
#                     max_llm_calls=int(150000),
#                     n_final_evals_per_prompt=500,
#                     baseline_accuracy=0.415,
#                     pruning=False)

# HPS_object = HPS(n_trials=2,
#                     llm_evaluation=llm_evaluation,
#                     llm_assistent=llm_assistent,
#                     dataset_name="math",
#                     seed=42,
#                     split=0.9,
#                     population_size=8,
#                     evaluations_per_prompt=25,
#                     max_evaluations_per_prompt=25,
#                     max_llm_calls=int(5000),
#                     n_final_evals_per_prompt=50,
#                     n_baseline_evals=100,
#                     n_testset_evals=1000,
#                     baseline_accuracy=0.4512,
#                     pruning=False)

HPS_object = HPS(n_trials=15,
                    llm_evaluation=llm_evaluation,
                    llm_assistent=llm_evaluation,
                    dataset_name="math",
                    seed=42,
                    split=0.9,
                    population_size=8,
                    evaluations_per_prompt=500,
                    max_evaluations_per_prompt=500,
                    max_llm_calls=int(150000),
                    n_final_evals_per_prompt=500,
                    n_testset_evals=5000,
                    baseline_accuracy=0.4512,
                    pruning=False)

HPS_object.run_HPS(previous_results_files=None)


