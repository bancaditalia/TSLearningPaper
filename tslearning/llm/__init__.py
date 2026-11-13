"""
The LLM comparison arm.

    client   provider-agnostic completion (ollama / openai / anthropic / file)
    prompt   what the model is shown and asked for
    propose  parse the completion into Rule objects and score them
    run      orchestration — see scripts/llm_rules.py

The whole point of the arrangement: the model is a rule PROPOSER, judged by the same
scoring code and the same train/freeze/test protocol as the template fitter. Nothing here
re-implements evaluation.
"""
