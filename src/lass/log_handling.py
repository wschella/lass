from __future__ import annotations
from typing import *
from pathlib import Path
import json

from tqdm import tqdm
import bigbench.api.results as bb

MODEL_FAMILIES = [
    "BIG-G T=0",
    "BIG-G T=1",
    "BIG-G sparse",
]

MODEL_SIZES = [
    "2m",
    "16m",
    "53m",
    "125m",
    "244m",
    "422m",
    "1b",
    "2b",
    "4b",
    "8b",
    "27b",
    "128b",
]

# Identify a trained LM by its model family + size.
ModelId = Tuple[str, str]

# The log for a specific model on a single task
# ModelLog  = bb.ResultsFileData

# The logs for multiple models on a single task
TaskLog = Dict[ModelId, bb.ResultsFileData]

TaskList = Union[
    Literal['paper-full'],
    Literal['paper-lite'],
    Literal['pipeline-test'],
    List[str]
]

# The different query types
QueryType = Union[Literal['generative'], Literal['multiple_choice'], Literal['scoring']]
QUERY_TYPES = {
    'generative': bb.GenerativeQuery,
    'multiple_choice': bb.MultipleChoiceQuery,
    'scoring': bb.ScoringQuery,
}

# The multiple types of query functions
QueryFunction = Union[Literal['cond_log_prob'], Literal['generate_text']]


class LogLoader():
    """
    Build a stream of tasks/queries/samples filtered by task/model/shots.

    After configuration, exposes 4 loading methods, each one level deeper:
    - per_task
        - per_model
            - per_query
                - per_sample
    """
    logdir: Path
    progress_bar: bool
    tasks: List[str]

    model_families: Optional[List[str]] = None
    model_sizes: Optional[List[str]] = None
    query_types: Optional[List[QueryType]] = None
    _query_types: Optional[List[bb.QueryType]] = None
    query_functions: Optional[List[QueryFunction]] = None
    shots: Optional[List[int]] = None
    include_unknown_shots: bool = False

    def __init__(
        self,
        logdir: Union[Path, str],
        tasks: Optional[Union[
            Literal['paper-full'],
            Literal['paper-lite'],
            Literal['pipeline-test'],
            List[str]
        ]] = None,
        model_families: Optional[List[str]] = None,
        model_sizes: Optional[List[str]] = None,
        query_types: Optional[List[QueryType]] = None,
        query_function: Optional[List[QueryFunction]] = None,
        shots: Optional[List[int]] = None,
        include_unknown_shots: bool = False,
        exclude_faulty_tasks: bool = True,
        progress_bar: bool = False
    ):
        self.progress_bar = progress_bar
        self.logdir = Path(logdir)
        if not self.logdir.exists():
            raise ValueError(f"Log directory {logdir} does not exist.")

        if tasks is None:
            tasks = 'paper-full'
        self.with_tasks(tasks, exclude_faulty=exclude_faulty_tasks)

        self.with_model_families(model_families)
        self.with_model_sizes(model_sizes)
        self.with_query_function(query_function)
        self.with_query_types(query_types)
        self.with_shots(shots, include_unknown=include_unknown_shots)

    def with_tasks(self, tasklist: TaskList, exclude_faulty: bool = True) -> LogLoader:
        if tasklist == 'paper-full':
            self.tasks = PaperTasks.full()
        elif tasklist == 'paper-lite':
            self.tasks = PaperTasks.lite()
        elif tasklist == 'pipeline-test':
            self.tasks = PaperTasks.lite()[:10]
        elif isinstance(tasklist, list):
            self.tasks = tasklist
        else:
            raise ValueError(f"Unknown tasklist: {tasklist}")

        if exclude_faulty:
            for task in LogIssues.with_issues():
                if task in self.tasks:
                    self.tasks.remove(task)

        return self

    def with_model_families(self, families: List[str] | None) -> LogLoader:
        self.model_families = families
        return self

    def with_model_sizes(self, sizes: List[str] | None) -> LogLoader:
        self.model_sizes = sizes
        return self

    def with_query_types(self, query_types: List[QueryType] | None) -> LogLoader:
        self.query_types = query_types
        if query_types is not None:
            self._query_types = [QUERY_TYPES[q] for q in query_types]
        return self

    def with_query_function(self, query_function: List[QueryFunction] | None) -> LogLoader:
        self.query_function = query_function
        return self

    def with_shots(self, shots: List[int] | None, include_unknown: bool = False) -> LogLoader:
        self.include_unknown_shots = include_unknown

        if shots is None:
            self.shots = shots
            return self

        # Copy here to avoid mutating the caller's list and making mypy angry.
        self.shots = [s for s in shots]

        return self

    def load_per_task(self) -> Iterator[TaskLog]:
        for logs in self._nested_results_generator():
            task: TaskLog = {}
            for model_id, log in logs:
                log = self._filter_queries(log)
                task[model_id] = log
            yield task

    def load_per_model(self) -> Iterator[bb.ResultsFileData]:
        for logs in self._nested_results_generator():
            for _, log in logs:
                log = self._filter_queries(log)
                yield log

    def load_per_query(self) -> Iterator[bb.QueryType]:
        for logs in self._nested_results_generator():
            for _, log in logs:
                for query in (log.queries or []):
                    if self._include_query(query):
                        yield query

    def load_per_sample(self) -> Iterator[bb.SampleType]:
        for logs in self._nested_results_generator():
            for _, log in logs:
                for query in (log.queries or []):
                    if self._include_query(query):
                        for sample in query.samples:
                            yield sample

    def _nested_results_generator(self) -> Iterator[Iterator[Tuple[ModelId, bb.ResultsFileData]]]:
        def __nested_generator(path) -> Iterator[Tuple[ModelId, bb.ResultsFileData]]:

            # Iterate over all models we care about.
            logfiles = (self.logdir / task).glob('*.json')
            for path in logfiles:

                # Filter out models we don't care about.
                model_family, model_size = self._extract_model_from_path(path)
                if self.model_families is not None and model_family not in self.model_families:
                    continue
                if self.model_sizes is not None and model_size not in self.model_sizes:
                    continue

                # Read and parse log file
                with path.open() as logfile:
                    try:
                        logs_json = json.load(logfile)
                        logs: bb.ResultsFileData = bb.ResultsFileData.fromdict(
                            logs_json, include_queries=True)
                    except Exception as e:
                        print(f"Failed to parse for task {task} at {path}")
                        raise e

                    yield ((model_family, model_size), logs)

        # Iterate over all tasks we care about.
        for task in tqdm(self.tasks, disable=not self.progress_bar):
            yield __nested_generator(task)

    def _filter_queries(self, results: bb.ResultsFileData) -> bb.ResultsFileData:
        results.queries = [q for q in (results.queries or []) if self._include_query(q)]
        return results

    def _include_query(self, query: bb.QueryType) -> bool:
        include = True
        if self.shots is not None:
            if (query.shots is not None) and (query.shots not in self.shots):
                include = False

        if self.include_unknown_shots is False:
            if query.shots is None:
                include = False

        if self._query_types is not None:
            if query.__class__ not in self._query_types:
                include = False

        if self.query_functions is not None:
            if query.function not in self.query_functions:
                include = False
        return include

    def _extract_model_from_path(self, path: Path) -> Tuple[str, str]:
        [_, model_family, model_size, *rest] = path.stem.split('_')
        if rest:
            model_family += ' ' + ' '.join(rest)
        return model_family, model_size


class LogIssues():
    @staticmethod
    def with_issues():
        """
        Returns a list of all tasks with some issues and might need to be avoided
        unless special care is taken.
        """
        return (
            LogIssues.with_different_samples_modelwise()
            + LogIssues.without_target()
        )

    @staticmethod
    def with_different_samples_modelwise():
        """
        Returns a list of tasks where the log files have different samples
        across different models.
        """
        return [
            "periodic_elements",  # different amount of queries
        ]

    @staticmethod
    def with_different_samples_shotwise():
        """
        Returns a list of tasks where the log files have different samples
        across different shots.
        Note: TODO: This is preliminary and not necessarily a problem.
        Some further investigation is needed to verify.
        """
        return [
            "anachronisms",
            "analogical_similarity",
            "analytic_entailment",
            "arithmetic",
            "authorship_verification",
            "causal_judgment",
            "cause_and_effect",
            "code_line_description",
            "common_morpheme",
            "conceptual_combinations",
            "crash_blossom",
            "crass_ai",
            "cryobiology_spanish",
            "cs_algorithms",
            "dark_humor_detection",
            "date_understanding",
            "disambiguation_qa",
            "discourse_marker_prediction",
            "dyck_languages",
            "emoji_movie",
            "emojis_emotion_prediction",
            "empirical_judgments",
            "english_proverbs",
            "english_russian_proverbs",
            "entailed_polarity",
            "entailed_polarity_hindi",
            "evaluating_information_essentiality",
            "fact_checker",
            "fantasy_reasoning",
            "figure_of_speech_detection",
            "general_knowledge",
            "geometric_shapes",
            "gre_reading_comprehension",
            "hhh_alignment",
            "hindu_knowledge",
            "hinglish_toxicity",
            "human_organs_senses",
            "identify_math_theorems",
            "identify_odd_metaphor",
            "implicatures",
            "implicit_relations",
            "intent_recognition",
            "international_phonetic_alphabet_nli",
            "irony_identification",
            "kanji_ascii",
            "kannada",
            "key_value_maps",
            "known_unknowns",
            "logic_grid_puzzle",
            "logical_args",
            "logical_deduction",
            "logical_sequence",
            "mathematical_induction",
            "metaphor_boolean",
            "metaphor_understanding",
            "minute_mysteries_qa",
            "misconceptions",
            "moral_permissibility",
            "movie_recommendation",
            "navigate",
            "nonsense_words_grammar",
            "novel_concepts",
            "odd_one_out",
            "penguins_in_a_table",
            "persian_idioms",
            "phrase_relatedness",
            "physical_intuition",
            "physics",
            "presuppositions_as_nli",
            "riddle_sense",
            "ruin_names",
            "salient_translation_error_detection",
            "sentence_ambiguity",
            "similarities_abstraction",
            "simple_ethical_questions",
            "snarks",
            "social_support",
            "sports_understanding",
            "strange_stories",
            "suicide_risk",
            "swahili_english_proverbs",
            "swedish_to_german_proverbs",
            "symbol_interpretation",
            "temporal_sequences",
            "understanding_fables",
            "undo_permutation",
            "unit_interpretation",
            "what_is_the_tao",
            "which_wiki_edit",
        ]

    @staticmethod
    def without_target():
        """
        Returns a list of tasks where the log files have no target included in
        the samples, causing parsing errors.
        """
        return [
            "abstraction_and_reasoning_corpus",
            "coqa_conversational_question_answering",
            "cycled_letters",
            "high_low_game",
            "multistep_arithmetic",
            "muslim_violence_bias",
            "program_synthesis",
            "python_programming_challenge",
            "question_answer_creation",
            "roots_optimization_and_games",
            "self_awareness",
            "self_evaluation_courtroom",
            "self_evaluation_tutoring",
            "spelling_bee",
            "squad_shifts",
            "sudoku",
            "taboo",
            "text_navigation_game",
            "truthful_qa",
            "twenty_questions",
            "word_problems_on_sets_and_graphs",
            "yes_no_black_white",
        ]


class PaperTasks():

    @staticmethod
    def full():
        """
        Returns the full list of tasks that have been used for the BIG-bench paper.
        Excludes 'training_on_test_set'.
        """
        return [
            "abstract_narrative_understanding",
            "abstraction_and_reasoning_corpus",
            "anachronisms",
            "analogical_similarity",
            "analytic_entailment",
            "arithmetic",
            "ascii_word_recognition",
            "authorship_verification",
            "auto_categorization",
            "auto_debugging",
            "bbq_lite",
            "bbq_lite_json",
            "bias_from_probabilities",
            "boolean_expressions",
            "bridging_anaphora_resolution_barqa",
            "causal_judgment",
            "cause_and_effect",
            "checkmate_in_one",
            "chess_state_tracking",
            "chinese_remainder_theorem",
            "cifar10_classification",
            "code_line_description",
            "codenames",
            "color",
            "com2sense",
            "common_morpheme",
            "conceptual_combinations",
            "conlang_translation",
            "context_definition_alignment",
            "coqa_conversational_question_answering",
            "crash_blossom",
            "crass_ai",
            "cryobiology_spanish",
            "cryptonite",
            "cs_algorithms",
            "cycled_letters",
            "dark_humor_detection",
            "date_understanding",
            "disambiguation_qa",
            "discourse_marker_prediction",
            "disfl_qa",
            "diverse_social_bias",
            "dyck_languages",
            "dynamic_counting",
            "elementary_math_qa",
            "emoji_movie",
            "emojis_emotion_prediction",
            "empirical_judgments",
            "english_proverbs",
            "english_russian_proverbs",
            "entailed_polarity",
            "entailed_polarity_hindi",
            "epistemic_reasoning",
            "evaluating_information_essentiality",
            "fact_checker",
            "factuality_of_summary",
            "fantasy_reasoning",
            "few_shot_nlg",
            "figure_of_speech_detection",
            "forecasting_subquestions",
            "formal_fallacies_syllogisms_negation",
            "gem",
            "gender_inclusive_sentences_german",
            "gender_sensitivity_chinese",
            "gender_sensitivity_english",
            "general_knowledge",
            "geometric_shapes",
            "goal_step_wikihow",
            "gre_reading_comprehension",
            "hhh_alignment",
            "high_low_game",
            "hindi_question_answering",
            "hindu_knowledge",
            "hinglish_toxicity",
            "human_organs_senses",
            "hyperbaton",
            "identify_math_theorems",
            "identify_odd_metaphor",
            "implicatures",
            "implicit_relations",
            "intent_recognition",
            "international_phonetic_alphabet_nli",
            "international_phonetic_alphabet_transliterate",
            "intersect_geometry",
            "irony_identification",
            "kanji_ascii",
            "kannada",
            "key_value_maps",
            "known_unknowns",
            "language_games",
            "language_identification",
            "linguistic_mappings",
            "linguistics_puzzles",
            "list_functions",
            "logic_grid_puzzle",
            "logical_args",
            "logical_deduction",
            "logical_fallacy_detection",
            "logical_sequence",
            "mathematical_induction",
            "matrixshapes",
            "metaphor_boolean",
            "metaphor_understanding",
            "minute_mysteries_qa",
            "misconceptions",
            "misconceptions_russian",
            "mnist_ascii",
            "modified_arithmetic",
            "moral_permissibility",
            "movie_dialog_same_or_different",
            "movie_recommendation",
            "mult_data_wrangling",
            "multiemo",
            "multistep_arithmetic",
            "muslim_violence_bias",
            "natural_instructions",
            "navigate",
            "nonsense_words_grammar",
            "novel_concepts",
            "object_counting",
            "odd_one_out",
            "operators",
            "paragraph_segmentation",
            "parsinlu_qa",
            "parsinlu_reading_comprehension",
            "penguins_in_a_table",
            "periodic_elements",
            "persian_idioms",
            "phrase_relatedness",
            "physical_intuition",
            "physics",
            "physics_questions",
            "play_dialog_same_or_different",
            "polish_sequence_labeling",
            "presuppositions_as_nli",
            "program_synthesis",
            "protein_interacting_sites",
            "python_programming_challenge",
            "qa_wikidata",
            "question_answer_creation",
            "question_selection",
            "real_or_fake_text",
            "reasoning_about_colored_objects",
            "repeat_copy_logic",
            "rephrase",
            "riddle_sense",
            "roots_optimization_and_games",
            "ruin_names",
            "salient_translation_error_detection",
            "scientific_press_release",
            "self_awareness",
            "self_evaluation_courtroom",
            "self_evaluation_tutoring",
            "semantic_parsing_in_context_sparc",
            "semantic_parsing_spider",
            "sentence_ambiguity",
            "similarities_abstraction",
            "simp_turing_concept",
            "simple_ethical_questions",
            "simple_text_editing",
            "snarks",
            "social_iqa",
            "social_support",
            "spelling_bee",
            "sports_understanding",
            "squad_shifts",
            "strange_stories",
            "strategyqa",
            "subject_verb_agreement",
            "sudoku",
            "sufficient_information",
            "suicide_risk",
            "swahili_english_proverbs",
            "swedish_to_german_proverbs",
            "symbol_interpretation",
            "taboo",
            "talkdown",
            "temporal_sequences",
            "tense",
            "text_navigation_game",
            "timedial",
            "topical_chat",
            "tracking_shuffled_objects",
            "truthful_qa",
            "twenty_questions",
            "understanding_fables",
            "undo_permutation",
            "unit_conversion",
            "unit_interpretation",
            "unnatural_in_context_learning",
            "unqover",
            "vitaminc_fact_verification",
            "web_of_lies",
            "what_is_the_tao",
            "which_wiki_edit",
            "winowhy",
            "word_problems_on_sets_and_graphs",
            "word_sorting",
            "word_unscrambling",
            "yes_no_black_white",
        ]

    @staticmethod
    def lite():
        """
        Returns the list of tasks that have been used in the "lite" set of tasks
        for the BIG-bench paper.
        """
        return [
            "auto_debugging",
            "bbq_lite_json",
            "code_line_description",
            "conceptual_combinations",
            "conlang_translation",
            "emoji_movie",
            "formal_fallacies_syllogisms_negation",
            "hindu_knowledge",
            "known_unknowns",
            "language_identification",
            "linguistics_puzzles",
            "logic_grid_puzzle",
            "logical_deduction",
            "misconceptions_russian",
            "novel_concepts",
            "operators",
            "parsinlu_reading_comprehension",
            "play_dialog_same_or_different",
            "repeat_copy_logic",
            "strange_stories",
            "strategyqa",
            "symbol_interpretation",
            "vitaminc_fact_verification",
            "winowhy",
        ]
