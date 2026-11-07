"""Frozen four-message English GSM8K prompt (v2)."""
V2_SYSTEM = "Solve the user's math problem step by step in English. End your response with a final line exactly in the form #### <number>, replacing <number> with the numeric answer. On that final line, write only #### followed by the number: no currency symbols, units, Markdown emphasis, or LaTeX."
V2_EXAMPLE_USER = 'What is 2 + 3?'
V2_EXAMPLE_ASSISTANT = '2 + 3 = 5.\n#### 5'
PROMPT_PROTOCOL = 'v2'

def build_messages_v2(question: str):
    """v2 four-message chat: system + format example + real question.
    The example is independent of GSM8K and never contains a target gold answer."""
    return [{'role': 'system', 'content': V2_SYSTEM}, {'role': 'user', 'content': V2_EXAMPLE_USER}, {'role': 'assistant', 'content': V2_EXAMPLE_ASSISTANT}, {'role': 'user', 'content': question}]

def render_prompt_v2(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        build_messages_v2(question),
        tokenize=False,
        add_generation_prompt=True,
    )

def count_prompt_tokens_v2(tokenizer, question: str) -> int:
    ids = tokenizer.apply_chat_template(
        build_messages_v2(question),
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(ids)
