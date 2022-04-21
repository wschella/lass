from __future__ import annotations
from tkinter import E
from typing import *
from pathlib import Path
import json

from tqdm import tqdm
import bigbench.api.results as bb

MODEL_FAMILIES = [
    "BIG-G T=0",
    "BIG-G T=1",
    "BIG-G-sparse",
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

Unit = Union[Literal['results-file'], Literal['query'], Literal['sample']]
QueryFunction = Union[Literal['cond_log_prob'], Literal['generate_text']]


class LogLoader():
    """
    Build a stream of tasks/queries/samples filtered by task/model/shots.
    """
    logdir: Path
    output_unit: Unit
    progress_bar: bool
    tasks: List[str]  # Not optional because we are opinionated here.

    model_families: Optional[List[str]] = None
    model_sizes: Optional[List[str]] = None
    query_types: Optional[List[Type[bb.QueryType]]] = None
    query_functions: Optional[List[QueryFunction]] = None
    shots: Optional[List[Union[int, None]]] = None

    def __init__(self, logdir: Path = Path('./artifacts/logs'), progress_bar: bool = False):
        self.progress_bar = progress_bar
        self.logdir = logdir

        self.output_unit = 'results-file'
        self.tasks = PaperTasks.full()

    def with_output_unit(self, unit: Unit) -> LogLoader:
        self.output_unit = unit
        return self

    def with_tasks(self, tasklist: Union[Literal['paper-full'], Literal['paper-lite'], List[str]]) -> LogLoader:
        if tasklist == 'paper-full':
            self.tasks = PaperTasks.full()
        elif tasklist == 'paper-lite':
            self.tasks = PaperTasks.lite()
        elif isinstance(tasklist, list):
            self.tasks = tasklist
        else:
            raise ValueError(f"Unknown tasklist: {tasklist}")
        return self

    def with_model_families(self, families: List[str]) -> LogLoader:
        self.model_families = families
        return self

    def with_model_sizes(self, sizes: List[str]) -> LogLoader:
        self.model_sizes = sizes
        return self

    def with_query_types(self, query_types: List[Type[bb.QueryType]]) -> LogLoader:
        self.query_types = query_types
        return self

    def with_query_function(self, query_function: List[QueryFunction]) -> LogLoader:
        self.query_function = query_function
        return self

    def with_shots(self, shots: List[int], include_unknown: bool = False) -> LogLoader:
        # Copy here to avoid mutating the caller's list and making mypy angry.
        self.shots = [s for s in shots]

        if include_unknown:
            self.shots.append(None)

        return self

    def load(self):
        """
        Returns an iterator over the tasks/queries/samples filtered by task/model/shots.
        Order/nesting is:
        - task
            - model
                - query
                    - sample
        """
        # Iterate over all tasks we care about.
        for task in tqdm(self.tasks, disable=not self.progress_bar):
            logfiles = (self.logdir / task).glob('*.json')

            # Iterate over all models we care about.
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

                    # Delegate yielding to specialised handlers
                    if self.output_unit == 'results-file':
                        yield from self._yield_for_task(logs)
                    if self.output_unit == 'query':
                        yield from self._yield_for_query(logs)
                    if self.output_unit == 'sample':
                        yield from self._yield_for_sample(logs)

    def _yield_for_task(self, results: bb.ResultsFileData) -> Iterator[bb.ResultsFileData]:
        if results.queries is not None:
            results.queries = [q for q in results.queries if self._include_query(q)]
        yield results

    def _yield_for_query(self, results: bb.ResultsFileData) -> Iterator[bb.QueryType]:
        assert results.queries is not None
        for query in results.queries:
            if self._include_query(query):
                yield query

    def _yield_for_sample(self, results: bb.ResultsFileData) -> Iterator[bb.SampleType]:
        assert results.queries is not None
        for query in results.queries:
            if self._include_query(query):
                for sample in query.samples:
                    yield sample

    def _include_query(self, query: bb.QueryType) -> bool:
        include = True
        if self.shots is not None:
            if query.shots not in self.shots:
                include = False

        if self.query_types is not None:
            if query.__class__ not in self.query_types:
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


class PaperTasks():

    @ staticmethod
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

    @ staticmethod
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
