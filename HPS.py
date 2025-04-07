import optuna
import logging
import datetime
import os
import re
from typing import List, Optional

from GA import GA
from utils.data_loader import DataLoader
from utils.eval_prompt import eval_prompt 


def save_plots(study, _):
    # Ensure the directory exists
    optimization_results_dir = f'{study.user_attrs["log_directory"]}/optimization_results'
    os.makedirs(optimization_results_dir, exist_ok=True)

    # Save best parameters
    with open(f'{optimization_results_dir}/best_params.txt', 'w') as f:
        f.write(str(study.best_params))

    # Save plots
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f'{optimization_results_dir}/optimization_history.png')
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(f'{optimization_results_dir}/slice.png')
    fig = optuna.visualization.plot_contour(study)
    fig.write_image(f'{optimization_results_dir}/contour.png')


def save_study_df_as_pkl(study, log_directory):
    results_dir = os.path.join(log_directory, 'optimization_results')
    os.makedirs(results_dir, exist_ok=True)
    
    df = study.trials_dataframe()
    df.to_pickle(os.path.join(results_dir, 'study_results.pkl'))
    print(f"Study trials DataFrame saved as PKL file at {os.path.join(results_dir, 'study_results.pkl')}")

def parse_previous_results(file_path):
    trials = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines) - 1):
            start_line = lines[i].strip()
            finish_line = lines[i + 1].strip()

            # Regex patterns to match the required parts
            start_pattern = re.compile(r'Start trial: (\d+), mutation_probability: ([0-9.]+), cross_over_ratio: ([0-9.]+)')
            finish_pattern = re.compile(r'Finish trial: (\d+), best_prompt: (.*?), accuracy: ([0-9.]+),')

            start_match = start_pattern.search(start_line)
            finish_match = finish_pattern.search(finish_line)

            if start_match and finish_match:
                try:
                    # Extracting the values from the regex match groups
                    trial_number = int(start_match.group(1))
                    mutation_probability = float(start_match.group(2))
                    cross_over_ratio = float(start_match.group(3))
                    best_prompt = finish_match.group(2).strip()
                    accuracy = float(finish_match.group(3))

                    trial = optuna.trial.FrozenTrial(
                        number=trial_number,
                        state=optuna.trial.TrialState.COMPLETE,
                        value=accuracy,
                        datetime_start=datetime.datetime.now(),
                        datetime_complete=datetime.datetime.now(),
                        params={'mutation_probability': mutation_probability, 'cross_over_ratio': cross_over_ratio},
                        distributions={
                            'mutation_probability': optuna.distributions.FloatDistribution(0.01, 1),
                            'cross_over_ratio': optuna.distributions.FloatDistribution(0.01, 1)
                        },
                        user_attrs={'best_prompt': best_prompt},
                        system_attrs={},
                        intermediate_values={},
                        trial_id=trial_number
                    )
                    trials.append(trial)
                except ValueError as e:
                    print(f"Error parsing lines: {start_line} or {finish_line}. Error: {e}")
                    continue
    return trials


def objective(trial, llm_evaluation, llm_assistent, dataset_name, seed, population, population_size, evaluations_per_prompt, max_evaluations_per_prompt,
              reset_evals_every_n_generations, n_final_evals_per_prompt, max_generations, max_llm_calls_per_run, baseline_accuracy, HPS_logger, HPS_directory,reset_evals_every_n_llm_calls,
              pruning):
        mutation_probability = trial.suggest_float("mutation_probability", 0.01, 1)
        cross_over_ratio = trial.suggest_float("cross_over_ratio", 0.01, 1)
        
        HPS_logger.info(f'objective(), Start trial: {trial.number}, mutation_probability: {mutation_probability}, cross_over_ratio: {cross_over_ratio}')
        GA_object = GA(llm_evaluation=llm_evaluation, llm_assistent=llm_assistent, dataset_name=dataset_name,seed=seed, population=population,
                                    max_generations=max_generations, evaluations_per_prompt=evaluations_per_prompt,
                                    max_evaluations_per_prompt=max_evaluations_per_prompt, n_final_evals_per_prompt=n_final_evals_per_prompt,
                                    population_size=population_size, reset_evals_every_n_generations=reset_evals_every_n_generations,
                                    baseline_accuracy=baseline_accuracy, mutation_probability=mutation_probability,
                                    cross_over_ratio=cross_over_ratio, max_llm_calls=max_llm_calls_per_run, GA_directory=HPS_directory,
                                    reset_evals_every_n_llm_calls=reset_evals_every_n_llm_calls, trial=trial, pruning=pruning)
        best_prompt, best_prompt_accuracy = GA_object.run_GA()
        trial.set_user_attr("best_prompt", best_prompt)
        HPS_logger.info(
                        f'objective(), Finish trial: {trial.number}, best_prompt: {best_prompt}, accuracy: {best_prompt_accuracy},'
                    )   

        return best_prompt_accuracy

class HPS:
    def __init__(self, n_trials, llm_evaluation, llm_assistent, dataset_name, seed=None, split=0.8, population: Optional[List[str]]=None, population_size=8,
                 evaluations_per_prompt=50, max_evaluations_per_prompt=100, n_final_evals_per_prompt=None,
                 reset_evals_every_n_generations=int(1e7), max_generations=int(1e7), max_llm_calls=int(1e10),
                 min_avrg_accuracy_increase_per_10_generations=-1.00, n_baseline_evals = 10000, n_testset_evals=10000, baseline_accuracy=0.41, reset_evals_every_n_llm_calls=int(1e7),
                 pruning=False):
        self.n_trials = n_trials
        self.llm_evaluation = llm_evaluation
        self.llm_assistent = llm_assistent
        self.dataset_name = dataset_name
        self.seed = seed
        self.split = split
        self.n_baseline_evals = n_baseline_evals
        self.n_testset_evals = n_testset_evals
        self.population_size = population_size
        self.population = population
        self.evaluations_per_prompt = evaluations_per_prompt
        self.max_evaluations_per_prompt = max_evaluations_per_prompt
        self.n_final_evals_per_prompt = n_final_evals_per_prompt
        self.reset_evals_every_n_generations = reset_evals_every_n_generations
        self.reset_evals_every_n_llm_calls = reset_evals_every_n_llm_calls
        self.max_generations = max_generations
        self.max_llm_calls = max_llm_calls
        self.max_llm_calls_per_run = self.max_llm_calls//self.n_trials
        self.min_avrg_accuracy_increase_per_10_generations = min_avrg_accuracy_increase_per_10_generations
        self.baseline_accuracy = baseline_accuracy
        self.pruning = pruning

        assert 0 < self.baseline_accuracy < 1, "Baseline accuracy is set to {self.baseline}. Should be between 0 and 1"

        self.HPS_logger, self.log_directory = self.setup_logger("HPS", "HPS.log")


    def setup_logger(self, name, log_file, level=logging.INFO, log_directory=None):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if log_directory is None:
            log_directory = f"./logs/{timestamp}"
        os.makedirs(log_directory, exist_ok=True)
        
        full_log_path = os.path.join(log_directory, log_file)
        formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
        handler = logging.FileHandler(full_log_path)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger, log_directory
    
    def log_parameter_HPO(self):
        self.HPS_logger.info(f'log_directory: {self.log_directory}')
        self.HPS_logger.info(f'dataset_name: {self.dataset_name}')
        self.HPS_logger.info(f'seed: {self.seed}')
        self.HPS_logger.info(f'split (train/test): {self.split}')
        self.HPS_logger.info(f'n_trials: {self.n_trials}')
        self.HPS_logger.info(f'population_size: {self.population_size}')
        self.HPS_logger.info(f'evaluations_per_prompt: {self.evaluations_per_prompt}')
        self.HPS_logger.info(f'max_evaluations_per_prompt: {self.max_evaluations_per_prompt}')
        self.HPS_logger.info(f'n_final_evals_per_prompt: {self.n_final_evals_per_prompt}')
        self.HPS_logger.info(f'reset_evals_every_n_generations: {self.reset_evals_every_n_generations}')
        self.HPS_logger.info(f'max_generations: {self.max_generations}')
        self.HPS_logger.info(f'max_llm_calls: {self.max_llm_calls}')
        self.HPS_logger.info(f'max_llm_calls_per_run: {self.max_llm_calls_per_run}')
        self.HPS_logger.info(f'min_avrg_accuracy_increase_per_10_generations: {self.min_avrg_accuracy_increase_per_10_generations}')
        self.HPS_logger.info(f'n_baseline_evals: {self.n_baseline_evals}')
        self.HPS_logger.info(f'n_testset_evals: {self.n_testset_evals}')
        self.HPS_logger.info(f'baseline_accuracy: {self.baseline_accuracy}')
        self.HPS_logger.info(f'reset_evals_every_n_llm_calls: {self.reset_evals_every_n_llm_calls}')
        self.HPS_logger.info(f'pruning enabled: {self.pruning}')


    def add_previous_trials(self, study, previous_results_file):
        previous_trials = parse_previous_results(previous_results_file)
        for trial in previous_trials:
            study.add_trial(trial)

    def run_HPS(self, previous_results_files=None):
        self.log_parameter_HPO()
        self.train_dataset = DataLoader(dataset_name=self.dataset_name, seed=self.seed, mode="train", split=self.split)
        self.test_dataset = DataLoader(dataset_name=self.dataset_name, seed=self.seed, mode="test", split=self.split)

        self.HPS_logger.info(f"Size train dataset: {len(self.train_dataset)}, Size test dataset: {len(self.test_dataset)}")
        self.HPS_logger.info(f"Size train dataset: {len(self.train_dataset)}, Size test dataset: {len(self.test_dataset)}")

        for i in range(min(3, len(self.train_dataset))):
            self.HPS_logger.info(f"Sample Training dataset {i}: {self.train_dataset[i]}")

        if self.baseline_accuracy is None:
            n = len(self.train_dataset) if self.n_baseline_evals is None else min(len(self.train_dataset), self.n_baseline_evals )
            self.baseline_accuracy = eval_prompt("", self.llm_evaluation, dataset=self.train_dataset, max_evals=n, logger=self.HPS_logger)
            self.HPS_logger.info(f'Baseline Accuracy = {self.baseline_accuracy}') 

        sampler = optuna.samplers.TPESampler()
        if self.pruning:
            study = optuna.create_study(direction="maximize", pruner=optuna.pruners.SuccessiveHalvingPruner(), sampler=sampler)
        else:
            study = optuna.create_study(direction="maximize", sampler=sampler)
        study.set_user_attr("log_directory", self.log_directory)

        if previous_results_files is not None:
            for previous_results_file in previous_results_files:
                self.add_previous_trials(study, previous_results_file)

        study.optimize(lambda trial: objective(
                                trial, self.llm_evaluation, self.llm_assistent, self.dataset_name, self.seed, self.population, self.population_size,
                                self.evaluations_per_prompt, self.max_evaluations_per_prompt, self.reset_evals_every_n_generations,
                                self.n_final_evals_per_prompt, self.max_generations, self.max_llm_calls_per_run, self.baseline_accuracy,
                                self.HPS_logger, self.log_directory, self.reset_evals_every_n_llm_calls, self.pruning
                                ),
                        n_trials=self.n_trials,
                        show_progress_bar=True,
                        # n_jobs=2,
                        timeout=None,
                        # Catch all errors
                        catch=(Exception,),
                        callbacks=[save_plots]
                        )
        save_study_df_as_pkl(study, self.log_directory)

        if len(study.best_trials) < 1:
            self.HPS_logger.warning("No successfull trials available to extract best prompt.")
            return None 
        best_prompt = study.best_trial.user_attrs.get("best_prompt", None)
        best_accuracy = study.best_value

        self.HPS_logger.info(f"Best Prompt: {best_prompt}, Best Accuracy (training set): {best_accuracy} ")

        for i in range(min(3, len(self.test_dataset))):
            self.HPS_logger.info(f"Sample Test dataset {i}: {self.test_dataset[i]}")

        test_logger, _ = self.setup_logger(name="EvalTestsetLogger", log_file="EvalTestset.log", level=logging.INFO, log_directory=self.log_directory)

        testset_acurracy = eval_prompt(best_prompt, self.llm_evaluation, self.test_dataset, max_evals=self.n_testset_evals, logger=test_logger)

        self.HPS_logger.info(f"Best Prompt: {best_prompt}, Best Accuracy (test set): {testset_acurracy}")

