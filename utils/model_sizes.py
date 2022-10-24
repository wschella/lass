from lass.log_handling import LogLoader, LoaderArgs

loader_args = LoaderArgs(
    logdir='./artifacts/logs',
    tasks=['abstract_narrative_understanding'],
    model_families=['BIG-G T=0'],
    # model_sizes=['128b'],
    query_types=['multiple_choice'],
    # shots=[0],
    include_unknown_shots=True,
    exclude_faulty_tasks=True,
)
loader = LogLoader(loader_args)

for log in loader.load_per_model():
    size = log.model.flop_matched_non_embedding_params
    print(f"\"{log.model.model_name}\": {size},")
