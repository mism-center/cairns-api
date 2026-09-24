import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_PROMPTS = {
    "REPHRASE_PROMPT": (
        "Rewrite the latest user question as a standalone question using chat history for missing context.\n"
        "Chat history:\n{chat_history}\n\n"
        "User question:\n{input}\n\n"
        "Return only the rewritten standalone question."
    ),
    "ANSWER_GENERATION_PROMPT": (
        "You are CAIRNS for ToolDB.\n"
        "Use only the provided <tools> context. Do not hallucinate facts.\n"
        "If the evidence is insufficient, explicitly say that no grounded answer is available.\n"
        "When you mention tools, cite the tool_id in square brackets, e.g. [tool_123].\n"
        "Only cite tool_ids that appear in context."
    ),
    "ANSWER_GENERATION_PROMPT_KG_APP": (
        "You answer ToolDB questions from graph evidence.\n"
        "Use only the provided <tools> context.\n"
        "If no grounded answer is available, say so clearly.\n"
        "Cite each referenced tool with [tool_id] from context only."
    ),
    "QV_KG_PROMPT": (
        "You are CAIRNS. Use only the provided <tools> evidence context.\n"
        "Incorporate user persona hints when present: {user_persona}\n"
        "Rules:\n"
        "1) Do not use external knowledge.\n"
        "2) If evidence is insufficient, say what is missing.\n"
        "3) Cite every referenced tool as [tool_id], and only if that tool_id exists in context."
    ),
    "QV_KG_SELECTION_PROMPT": (
        "You are CAIRNS. Select grounded ToolDB options using only the provided <tools> evidence context.\n"
        "Incorporate user persona hints when present: {user_persona}\n"
        "User requested up to {requested_count} options.\n\n"
        "Evidence context:\n{context}\n\n"
        "Return strict JSON only with keys:\n"
        "- selected_tool_ids (array of tool_id strings)\n"
        "- missing_constraints (array of strings)\n"
        "Rules:\n"
        "1) Select at most {requested_count} tool_ids.\n"
        "2) Only use tool_ids that appear in the evidence context.\n"
        "3) Prefer tools that best satisfy the user request.\n"
        "4) If the evidence does not clearly support some requested constraints, list them in missing_constraints.\n"
        "5) Return JSON only."
    ),
    "INTENT_PROMPT": (
        "Extract user intent from the conversation.\n"
        "Return strict JSON only with keys:\n"
        "- user_goal (string)\n"
        "- constraints (array of strings)\n"
        "- output_preference (string)\n"
        "- confidence (number between 0 and 1)"
    ),
    "QUERY_RELEVANCE_CLASSIFIER_PROMPT": (
        "Classify the user request for routing.\n"
        "Return exactly one token:\n"
        "- documentation: if user asks what the bot/system can do, setup, usage, or docs\n"
        "- lookup: for tool/data retrieval or research questions\n"
        "Return only `documentation` or `lookup`."
    ),
    "DOCUMENTATION_GENERATION": (
        "You are generating a concise capability summary.\n"
        "Use only the provided repository description below.\n\n"
        "Repository description:\n{documentation}\n\n"
        "User request:\n{input}\n\n"
        "Return a concise answer."
    ),
    "GUARDRAILS_CONFIG": "",
    "INPUT_GUARDRAILS": "",
}


def _load_prompts_file(path: str) -> dict[str, str]:
    prompt_file = Path(path)
    if not prompt_file.exists():
        return {}
    with prompt_file.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        return {}
    output = {}
    for key, value in raw.items():
        if isinstance(value, str):
            output[key] = value
    return output


def get_prompt_text(prompt_name: str, config: Any, langfuse_client: Optional[Any] = None) -> str:
    if getattr(config, "LANGFUSE_ENABLED", False):
        try:
            client = langfuse_client or getattr(config, "langfuse", None)
            if client is not None:
                prompt = client.get_prompt(prompt_name)
                if prompt and getattr(prompt, "prompt", None):
                    return prompt.prompt
        except Exception:
            pass

    prompts_file = str(getattr(config, "PROMPTS_FILE", "") or "").strip()
    if prompts_file:
        try:
            file_prompts = _load_prompts_file(prompts_file)
            if prompt_name in file_prompts:
                return file_prompts[prompt_name]
        except Exception:
            pass

    if prompt_name in DEFAULT_PROMPTS:
        return DEFAULT_PROMPTS[prompt_name]

    raise KeyError(f"Prompt '{prompt_name}' not found in Langfuse, PROMPTS_FILE, or built-in defaults.")
