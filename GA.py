import matplotlib
matplotlib.use('Agg')

import datetime
import logging
import os
import pandas as pd
import random
from utils.nlp import sample_word, make_valid_instruction
from utils.utils import dict_has_at_least_two_nonzero_values
from utils.utils import evaluate_single_example
from utils.eval_prompt import eval_prompt
from utils.data_loader import DataLoader
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import optuna


class GA:
    def __init__(self, llm_evaluation, llm_assistent, dataset_name, seed=None, split=0.9, population: Optional[List[str]]=None, population_size=8,
                 evaluations_per_prompt=50, max_evaluations_per_prompt=100, n_final_evals_per_prompt=None,
                 mutation_probability=0.2, cross_over_ratio=0.8, reset_evals_every_n_generations=int(1e7),
                 max_generations=int(1e7), max_llm_calls=int(1e10), min_avrg_accuracy_increase_per_10_generations=-1.00,
                 baseline_accuracy=0.41, GA_directory=None, reset_evals_every_n_llm_calls=(1e10), n_testset_evals=1000, trial=None, pruning=False):
        self.llm_evaluation = llm_evaluation
        self.llm_assistent = llm_assistent
        self.dataset_name = dataset_name
        if population is not None:
            self.population_size = len(population)
        else:
            self.population_size = population_size
        self.population = population
        self.evaluations_per_prompt = evaluations_per_prompt
        self.max_evaluations_per_prompt = max_evaluations_per_prompt
        self.n_final_evals_per_prompt = n_final_evals_per_prompt
        assert self.n_final_evals_per_prompt != 0, "n_final_evals_per_prompt should be positive integer or None."
        self.mutation_probability = mutation_probability
        self.cross_over_ratio = cross_over_ratio
        self.reset_evals_every_n_generations = reset_evals_every_n_generations
        assert self.reset_evals_every_n_generations != 0, "reset_evals_every_n_generations should be positive integer."
        self.reset_evals_every_n_llm_calls = reset_evals_every_n_llm_calls
        assert self.reset_evals_every_n_llm_calls != 0, "reset_evals_every_n_llm_calls should be positive integer."
        self.last_checkpoint = 0
        self.max_generations = max_generations
        self.max_llm_calls = max_llm_calls
        assert self.max_llm_calls > self.evaluations_per_prompt * self.population_size, "max_llm_calls should be greater than evaluations_per_prompt * population_size."
        self.min_avrg_accuracy_increase_per_10_generations = min_avrg_accuracy_increase_per_10_generations
        self.GA_directory = GA_directory
        self.ga_logger, self.log_directory = self.setup_logger("GA_logger", "ga.log", GA_folder=self.GA_directory)
        self.llm_logger, _ = self.setup_logger("LLM_logger", "llm.log", GA_folder=self.GA_directory)
        self.baseline_accuracy = baseline_accuracy
        if population is not None:
                self.current_population = {prompt: [0, 0] for prompt in self.population}
        else:
            self.current_population = {}
            self.initiate_population()
        self.old_population = {}
        self.history_average_accuracy = [] # List of float
        self.history_best_prompt = [] # List of str
        self.history_best_prompt_accuracy = [] # List of float
        self.history_best_pos_evals = [] # List of int
        self.history_best_neg_evals = [] # List of int
        self.generation_counter = 0
        self.fitness_factor_baseline = 0.8
        self.fitness_exponential_factor = 2
        self.print_max_chars_prompt = 80
        self.trial = trial
        self.pruning = pruning
        self.n_testset_evals = n_testset_evals
        self.seed = seed
        self.split = split

        self.train_dataset = DataLoader(dataset_name=self.dataset_name, seed=self.seed, mode="train", split=self.split)
        self.test_dataset = DataLoader(dataset_name=self.dataset_name, seed=self.seed, mode="test", split=self.split)

        self.ga_logger.info(f"Size train dataset: {len(self.train_dataset)}, Size test dataset: {len(self.test_dataset)}")
        self.ga_logger.info(f"Size train dataset: {len(self.train_dataset)}, Size test dataset: {len(self.test_dataset)}")

    def plot_history(self):
        os.makedirs(self.log_directory, exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(self.history_average_accuracy, label='Average Accuracy')
        plt.plot(self.history_best_prompt_accuracy, label='Best Accuracy')
        plt.title('Generation vs Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plot_filename = os.path.join(self.log_directory, 'accuracy_over_generations.png')
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        plt.close()

    def initiate_population(self):
        for i in range(self.population_size):
            # Sample the prompt length
            prompt_length = random.randint(2, 10)
            # Sample words
            words = [sample_word() for _ in range(prompt_length)]
            prompt = " ".join(words)
            prompt = make_valid_instruction(prompt, self.llm_assistent, self.llm_logger)
            self.add_prompt_to_current_population(prompt)
    
    def setup_logger(self, name, log_file, level=logging.INFO, log_directory=None, GA_folder=None):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if log_directory is not None:
            raise ValueError("log_directory should be None; use GA_folder instead.")

        # Set log_directory based on GA_folder or default to ./logs/
        if GA_folder is None:
            log_directory = f"./logs/{timestamp}"
        else:
            log_directory = f"{GA_folder}/{timestamp}"
        os.makedirs(log_directory, exist_ok=True)
        logger = logging.getLogger(name)
        
        # If the logger already has handlers, remove them to reset the logger
        if logger.hasHandlers():
            for handler in logger.handlers:
                logger.removeHandler(handler)

        full_log_path = os.path.join(log_directory, log_file)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        handler = logging.FileHandler(full_log_path)
        handler.setFormatter(formatter)

        logger.setLevel(level)
        logger.addHandler(handler)

        return logger, log_directory

    def add_prompt_to_current_population(self, prompt):
        if prompt not in self.current_population:   
            self.current_population[prompt] = [0, 0]
        else:
            self.log_and_print_message(f"Prompt '{prompt}' already exists in the current population. Skipping addition.")
    
    def increase_pos_eval_of_current_gen_by_one(self, prompt):
        self.current_population[prompt][0] += 1

    def increase_neg_eval_of_current_gen_by_one(self, prompt):
        self.current_population[prompt][1] += 1

    def get_accuracy(self, prompt):
        population = self.current_population if prompt in self.current_population else self.old_population
        if sum(population[prompt]) == 0:
            return 0
        return population[prompt][0] / sum(population[prompt])

    def move_prompt_from_current_to_old_population(self, prompt):
        self.old_population[prompt] = self.current_population.pop(prompt)
        self.ga_logger.info(f"Moved prompt from current to old population: {prompt}")

    def format_prompt(self, prompt, values):
        """Helper function to format prompt output."""
        formatted_prompt = prompt[:self.print_max_chars_prompt].ljust(self.print_max_chars_prompt)
        accuracy = self.get_accuracy(prompt)
        return f"Prompt: {formatted_prompt}, pos_eval: {values[0]:5}, neg_eval: {values[1]:5}, accuracy: {accuracy:.4f}"

    def print_population(self, population, message=None):
        """Generic function to print any population details."""
        if message:
            print(message)
        print(f"Population size: {len(population)}")
        for prompt, values in population.items():
            print(self.format_prompt(prompt, values))

    def log_population(self, population, message=None):
        """Generic function to log any population details."""
        if message:
            self.ga_logger.info(message)
        for prompt, values in population.items():
            self.ga_logger.info(self.format_prompt(prompt, values))

    def print_current_population(self, message=None):
        self.print_population(self.current_population, message)

    def print_old_population(self, message=None):
        self.print_population(self.old_population, message)

    def log_current_population(self, message=None):
        self.sort_current_population_by_accuracy()
        self.log_population(self.current_population, message)

    def log_old_population(self, message=None):
        self.sort_old_population_by_accuracy()
        self.log_population(self.old_population, message)

    def save_history_as_pickeld_df(self):
        data = {
            'Average Accuracy': self.history_average_accuracy,
            'Best Prompt': self.history_best_prompt,
            'Best Accuracy': self.history_best_prompt_accuracy,
            'Best Pos Evals': self.history_best_pos_evals,
            'Best Neg Evals': self.history_best_neg_evals
        }
        df = pd.DataFrame(data)
        df.index.name = 'Generation'
        filename = 'generation_data.pkl'
        file_path = os.path.join(self.log_directory, filename)
        df.to_pickle(file_path)
        self.ga_logger.info(f"DataFrame saved as pickle to {file_path}")


    def stop_criteria_met(self) -> bool:
        if self.llm_evaluation.counter_llm_calls >= self.max_llm_calls:
            self.ga_logger.info(f"Max LLM calls reached. Counter: {self.llm_evaluation.counter_llm_calls}. Max: {self.max_llm_calls}")
            return True
        if self.generation_counter > self.max_generations:
            self.ga_logger.info(f"Max generations reached. Counter: {self.generation_counter}. Max: {self.max_generations}")
            return True
        if self.generation_counter > 10:
            average_accuracy_last_5_gen = sum(self.history_average_accuracy[-5:]) / 5
            average_accuracy_previous_5_gen = sum(self.history_average_accuracy[-10:-5]) / 5

            #increase_percentage = (average_accuracy_last_5_gen - average_accuracy_previous_5_gen) / average_accuracy_previous_5_gen
            if average_accuracy_previous_5_gen == 0:
                increase_percentage = float('inf')
            else:
                increase_percentage = (average_accuracy_last_5_gen - average_accuracy_previous_5_gen) / average_accuracy_previous_5_gen

            if increase_percentage < self.min_avrg_accuracy_increase_per_10_generations:
                self.ga_logger.info(f"Average accuracy increase per last 5 generations over previous 5 generations too low. Increase in percent: {increase_percentage:.2f}")
                return True
        else:
            self.ga_logger.info("No stop criteria met.")
            return False
    
    def reset_criteria_met(self) -> bool:
        if self.reset_evals_every_n_llm_calls == None:
            return False
        i = self.llm_evaluation.counter_llm_calls
        n_calls = self.reset_evals_every_n_llm_calls
        print(f'i: {i}, n_calls: {n_calls}, last_checkpoint: {self.last_checkpoint}')
        reset = i // n_calls > self.last_checkpoint // n_calls
        self.ga_logger.info(f"reset_criteria_met(), True or False: {reset}, counter_llm_calls: {i}, reset_every_n_llm_calls: {self.reset_evals_every_n_llm_calls}, "
                            f"counter_llm_calls // reset_every_n_llm_calls: {i // n_calls}, last_checkpoint: {self.last_checkpoint}")
        if reset:
            self.last_checkpoint = (i // n_calls) * n_calls
            self.ga_logger.info(f"Resetting evaluations in generation {self.generation_counter} after {i} LLM calls.")
            return True
        return False

    def evaluate_problem(self, prompt):
        index_random_problem = random.randint(0, len(self.train_dataset)-1)
        problem = self.train_dataset[index_random_problem]
        question = problem['question']
        solution = problem['answer']
        correct = evaluate_single_example(question, solution, prompt, self.llm_evaluation, self.llm_logger)

        if correct:
            self.increase_pos_eval_of_current_gen_by_one(prompt)
        else:
            self.increase_neg_eval_of_current_gen_by_one(prompt)
        
        formatted_answer = self.llm_evaluation.predict(f"{question} {prompt}", self.llm_logger, max_tokens=1000).replace('\\n', '. ')
        self.llm_logger.info(f"Prompt: {prompt[:self.print_max_chars_prompt]:.{self.print_max_chars_prompt}}, Question: {question}, Solution: {solution}, Answer: {formatted_answer}, Correct: {correct}")
        
    def calc_fitness(self, accuracy):
        if accuracy < self.baseline_accuracy * self.fitness_factor_baseline:
            return 0.
        else:
            scaled_accuracy = (accuracy - self.baseline_accuracy * self.fitness_factor_baseline)/(1 - self.baseline_accuracy * self.fitness_factor_baseline)
            fitness = scaled_accuracy**self.fitness_exponential_factor
            return float(fitness)
        
    def get_total_evaluations_per_prompt(self, prompt):
        return self.current_population[prompt][0] + self.current_population[prompt][1]

    def sort_current_population_by_accuracy(self):
        self.current_population = dict(sorted(self.current_population.items(), key=lambda item: self.get_accuracy(item[0]), reverse=True))
    
    def sort_old_population_by_accuracy(self):
        self.old_population = dict(sorted(self.old_population.items(), key=lambda item: self.get_accuracy(item[0]), reverse=True))

    def evaluate_current_population(self, n_evals=None):
            skip_max_check = True if n_evals is not None else False

            n_evals = n_evals if n_evals is not None else self.evaluations_per_prompt

            tasks = [
                (prompt, n_evals)
                for prompt in self.current_population
                if skip_max_check or (self.get_total_evaluations_per_prompt(prompt) < self.max_evaluations_per_prompt)
            ]

            def perform_evaluations(prompt, times):
                for _ in range(times):
                    self.evaluate_problem(prompt)

            with ThreadPoolExecutor() as executor:
                executor.map(lambda args: perform_evaluations(*args), tasks)

            self.log_current_population(f'Generation: {self.generation_counter} after evaluation')

    def select_parents(self, population_parents_probabilities) -> List[str]:
        best_parents = []
        for i in range(2):            
            best_parent = random.choices(list(population_parents_probabilities.keys()), weights=list(population_parents_probabilities.values()), k=1)[0]
            best_parents.append(best_parent)
            population_parents_probabilities.pop(best_parent)
        self.ga_logger.info(f"Selected parents: {best_parents}")
        return best_parents
    
    def single_crossover(self, selected_parents) -> List[str]:
        offspring = []

        parent1 = selected_parents[0]
        parent2 = selected_parents[1]

        words1 = parent1.split()
        words2 = parent2.split()

        split_point_1 = random.randint(1, len(words1)-1)
        split_point_2 = random.randint(1, len(words2)-1)

        offspring_1 = " ".join(words1[:split_point_1] + words2[split_point_2:])
        offspring_2 = " ".join(words2[:split_point_2] + words1[split_point_1:])

        offspring_1 = make_valid_instruction(offspring_1, self.llm_assistent, self.llm_logger)
        offspring_2 = make_valid_instruction(offspring_2, self.llm_assistent, self.llm_logger)

        offspring.append(offspring_1)
        offspring.append(offspring_2)

        return offspring

    def crossover(self):
        self.ga_logger.info(f"crossover(), generation: {self.generation_counter}, population size before crossover: {len(self.current_population)}")
        population_parents_probabilities = {}
        for prompt in self.current_population:
            accuracy = self.get_accuracy(prompt)
            fitness = self.calc_fitness(accuracy)
            population_parents_probabilities[prompt] = fitness
        self.ga_logger.info(f"Population parents probabilities before crossover: {population_parents_probabilities}")
        offspring_crossover = []
        count = int(self.cross_over_ratio * (self.population_size /2))
        count = max(1, count)
        count = min(count, self.population_size //2)

        self.ga_logger.info(f"Count of crossovers: {count}")

        for i in range(count):
            self.ga_logger.info(f"Crossover {i+1}/{count}")
            # Check if there are at least two prompts with probabilities > 0 to select as parents
            if dict_has_at_least_two_nonzero_values(population_parents_probabilities):
                selected_parents = self.select_parents(population_parents_probabilities)
            else:
                self.ga_logger.info("Not enough parents to select from.")
                break
            offspring = self.single_crossover(selected_parents)
            offspring_crossover.extend(offspring)
        self.ga_logger.info(f"Offspring from crossover: {offspring_crossover}")
        self.ga_logger.info(f"Population parents probabilities after crossover: {population_parents_probabilities}")
        for prompt in offspring_crossover:
            self.add_prompt_to_current_population(prompt)
        self.print_and_log_current_population(f"Generation: {self.generation_counter} after crossover")

    def mutation(self):
        mutated_prompts = []
        new_prompts = []
        counter_mutations = 0

        # Only up to population_size to avoid adding prompts from crossover
        for prompt in list(self.current_population)[:self.population_size]: 
            if random.random() < self.mutation_probability:
                words = prompt.split()
                word_to_replace = random.choice(words)
                new_word = sample_word()
                # Replace word in the list
                words[words.index(word_to_replace)] = new_word  
                mutated_prompt = " ".join(words)
                mutated_prompt = make_valid_instruction(mutated_prompt, self.llm_assistent, self.llm_logger)
                mutated_prompts.append(mutated_prompt)
                new_prompts.append(mutated_prompt)  
                counter_mutations += 1
                self.llm_logger.info(f"Mutation: {prompt[:self.print_max_chars_prompt]:.{self.print_max_chars_prompt}}"
                                     f" -> {mutated_prompt[:self.print_max_chars_prompt]:.{self.print_max_chars_prompt}}")

        for prompt in new_prompts:
            self.add_prompt_to_current_population(prompt)

        self.ga_logger.info(f"Added {counter_mutations} mutated prompts to current generation. Mutated prompts: {mutated_prompts}")
        self.print_and_log_current_population(f"Generation: {self.generation_counter} after mutation")

    def selection(self):
        self.ga_logger.info(f"selection(), generation: {self.generation_counter}, population size before selection: {len(self.current_population)}")    
        # Keep the best prompts up to population size, move the rest to old population
        self.sort_current_population_by_accuracy()
        best_prompts = list(self.current_population.keys())[:self.population_size]
    
        prompts_to_move = [
            prompt for prompt in self.current_population
            if prompt not in best_prompts
        ]
        for prompt in prompts_to_move:
            self.move_prompt_from_current_to_old_population(prompt)

    def calc_average_accuracy_current_population(self):
        return sum([self.get_accuracy(prompt) for prompt in self.current_population]) / len(self.current_population)

    def log_current_population_to_lists(self):
        self.sort_current_population_by_accuracy()
        best_prompt = list(self.current_population.keys())[0]
        self.history_best_prompt.append(best_prompt)
        best_prompt_accuracy = self.get_accuracy(best_prompt)
        self.history_best_prompt_accuracy.append(best_prompt_accuracy)
        self.history_average_accuracy.append(self.calc_average_accuracy_current_population())
        self.history_best_pos_evals.append(self.current_population[best_prompt][0])
        self.history_best_neg_evals.append(self.current_population[best_prompt][1])

    def log_parameters_GA(self):
        self.ga_logger.info(f'population_size: {self.population_size}')
        self.ga_logger.info(f'evaluations_per_prompt: {self.evaluations_per_prompt}')
        self.ga_logger.info(f'max_evaluations_per_prompt: {self.max_evaluations_per_prompt}')
        self.ga_logger.info(f'mutation_probability: {self.mutation_probability}')
        self.ga_logger.info(f'cross_over_ratio: {self.cross_over_ratio}')
        self.ga_logger.info(f'reset_evals_every_n_generations: {self.reset_evals_every_n_generations}')
        self.ga_logger.info(f'reset_evals_every_n_llm_calls: {self.reset_evals_every_n_llm_calls}')
        self.ga_logger.info(f'n_final_evals_per_prompt: {self.n_final_evals_per_prompt}')
        self.ga_logger.info(f'max_generations: {self.max_generations}')
        self.ga_logger.info(f'max_llm_calls: {self.max_llm_calls}')
        self.ga_logger.info(f'min_avrg_accuracy_increase_per_10_generations: {self.min_avrg_accuracy_increase_per_10_generations}')
        self.ga_logger.info(f'baseline_accuracy: {self.baseline_accuracy}')
        self.ga_logger.info(f'fitness_factor_baseline: {self.fitness_factor_baseline}')
        self.ga_logger.info(f'fitness_exponential_factor: {self.fitness_exponential_factor}')
        self.ga_logger.info(f'generation_counter: {self.generation_counter}')


    def print_and_log_current_population(self, message):
        self.print_current_population(message)
        self.log_current_population(message)

    def log_and_store_data_current_population(self, message=""):
        self.log_current_population_to_lists()
        self.print_and_log_current_population(f'Generation: {self.generation_counter}'+ " " + message)
        self.plot_history()
        self.save_history_as_pickeld_df()

    def log_and_print_message(self, message):
        self.ga_logger.info(message)
        print(message)

    def log_history_best_prompts(self):
        self.ga_logger.info("Best prompts and their final accuracy:")
        for i in range(len(self.history_best_prompt)):
            prompt = self.history_best_prompt[i]
            accuracy = self.history_best_prompt_accuracy[i]
            pos_evals = self.history_best_pos_evals[i]
            neg_evals = self.history_best_neg_evals[i]
            
            self.ga_logger.info(
                f"Generation: {i}, "
                f"Prompt: {prompt[0:self.print_max_chars_prompt]:.{self.print_max_chars_prompt}}, "
                f"Positive evaluations: {pos_evals}, "
                f"Negative evaluations: {neg_evals}, "
                f"Accuracy: {accuracy:.4f}"
            )

    def reset_evaluations_in_current_population(self):
        for prompt in self.current_population:
            self.current_population[prompt] = [0, 0]

    def log_and_counter_at_end_of_generation(self):
        self.log_and_store_data_current_population()
        self.log_history_best_prompts()
        self.generation_counter += 1
        self.log_and_print_message('Current LLM evaluation calls: ' + str(self.llm_evaluation.counter_llm_calls) + ' of ' + str(self.max_llm_calls))

    def run_GA(self):
        self.llm_evaluation.reset_call_counter()
        self.log_parameters_GA()

        try:
            while not self.stop_criteria_met():
                self.log_and_print_message(f'---------------------------- Start Generation: {self.generation_counter} ----------------------------')

                if self.generation_counter == 0:
                    self.evaluate_current_population()
                    self.log_and_counter_at_end_of_generation()
                    continue

                self.crossover()
                self.mutation()
                self.evaluate_current_population()
                self.selection()

                # Reset evaluations every n generations
                if self.generation_counter % self.reset_evals_every_n_generations == 0:
                    self.reset_evaluations_in_current_population()
                    self.evaluate_current_population()
                    self.ga_logger.info(f"Reset evaluations in generation {self.generation_counter} due to reset_evals_every_n_generations.")

                # Reset evaluations every n llm calls
                if self.reset_criteria_met():
                    self.reset_evaluations_in_current_population()
                    self.evaluate_current_population()
                    self.ga_logger.info(f"Reset evaluations in generation {self.generation_counter} due to reset_evals_every_n_llm_calls.")

                    # Pruning only after reset to prun based on unbiased results
                    if self.pruning:
                        self.trial.report(self.get_accuracy(self.history_best_prompt[-1]), self.generation_counter)
                        if self.trial.should_prune():
                            self.ga_logger.info(f'Pruning at generation {self.generation_counter}')
                            raise optuna.TrialPruned()  

                self.log_and_counter_at_end_of_generation()
        except KeyboardInterrupt:
            self.log_and_print_message(f"Algorithm interrupted during generation {self.generation_counter} by user. Proceeding to next phase.")

        try:
            # If the final population has not been evaluated up to max_evaluations_per_prompt, evaluate it
            if self.evaluations_per_prompt < self.max_evaluations_per_prompt:
                self.log_and_print_message(f'Final population before evaluation until max evaluations pro prompt of {self.max_evaluations_per_prompt}')
                self.print_current_population()

                # call evaluation multiple times, such that all prompts are evaluated up to self.max_evaluations_per_prompt
                rounds = self.max_evaluations_per_prompt // self.evaluations_per_prompt
                [self.evaluate_current_population() for i in range(rounds)]
        except KeyboardInterrupt:
            self.log_and_print_message("Algorithm interrupted during evaluating up to max_evaluations_per_prompt by user. Proceeding to next phase.")

        print(f'Final population:')
        self.sort_current_population_by_accuracy()
        self.sort_old_population_by_accuracy()
        self.print_current_population()
        self.print_old_population()

        self.log_current_population('----- Final current_population -----')
        self.log_old_population('----- Final old_population -----')

        try:
            # Reset and evaluate the final population if there is a n_final_evals_per_prompt
            if self.n_final_evals_per_prompt is not None:
                self.reset_evaluations_in_current_population()
                self.evaluate_current_population(n_evals=self.n_final_evals_per_prompt)
                self.sort_current_population_by_accuracy()
                self.print_current_population(message='----- Population after reset and final evaluation -----')
                self.log_and_store_data_current_population(message='----- Population after reset and final evaluation -----')
        except KeyboardInterrupt:
            self.log_and_print_message("Algorithm interrupted during evaluations according to n_final_evals_per_prompt. Proceeding to next phase.")

        self.log_and_print_message('Total LLM evaluation calls: ' + str(self.llm_evaluation.counter_llm_calls))

        testset_acurracy = eval_prompt(self.history_best_prompt[-1], self.llm_evaluation, self.test_dataset, max_evals=self.n_testset_evals, logger=self.llm_logger)

        self.ga_logger.info(f"Best Prompt: {self.history_best_prompt[-1]}, Best Accuracy (test set): {testset_acurracy}")

        return self.history_best_prompt[-1], self.history_best_prompt_accuracy[-1] 