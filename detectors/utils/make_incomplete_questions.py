from dotenv import load_dotenv
import os
import json
import argparse
import requests
from pathlib import Path
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Get OpenRouter API key from environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "6"))
OPENROUTER_RETRY_BASE_SECONDS = float(os.getenv("OPENROUTER_RETRY_BASE_SECONDS", "1.0"))
OPENROUTER_RETRY_MAX_SECONDS = float(os.getenv("OPENROUTER_RETRY_MAX_SECONDS", "30.0"))
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "90"))
OPENROUTER_INCOMPLETE_WORKERS = int(os.getenv("OPENROUTER_INCOMPLETE_WORKERS", "8"))


def _sleep_with_backoff(attempt: int) -> None:
    base = min(OPENROUTER_RETRY_MAX_SECONDS, OPENROUTER_RETRY_BASE_SECONDS * (2 ** attempt))
    jitter = 0.5 + random.random()  # [0.5, 1.5)
    time.sleep(base * jitter)


def _openrouter_chat_completion_content(*, headers: dict, data: dict) -> str:
    last_err = None
    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=OPENROUTER_TIMEOUT_SECONDS,
            )
            if response.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
                raise RuntimeError(f"transient HTTP {response.status_code}: {response.text[:2000]}")
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:2000]}")
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty/non-string OpenRouter content")
            return content
        except Exception as e:
            last_err = e
            if attempt >= OPENROUTER_MAX_RETRIES:
                raise
            _sleep_with_backoff(attempt)
    raise RuntimeError(f"OpenRouter call failed: {last_err}")

def load_questions_from_file(input_file):
    """Load questions from a JSON or JSONL file. Returns list of dicts with 'question' and optionally 'dataset'."""
    questions = []
    
    with open(input_file, 'r') as f:
        # Try to parse as JSON first
        try:
            f.seek(0)
            data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # Check if this is similar_questions.json format (has dataset field)
                if data and isinstance(data[0], dict) and "dataset" in data[0]:
                    # Handle similar_questions.json structure
                    for entry in data:
                        dataset = entry.get("dataset")
                        original_question_id = entry.get("original_question_id")
                        
                        # Add original question with dataset
                        if "original_question" in entry:
                            questions.append({
                                "question": entry["original_question"],
                                "dataset": dataset,
                                "original_question_id": original_question_id,
                                "type": "original"
                            })
                        
                        # Add similar questions with dataset
                        if "similar_questions" in entry:
                            for sq in entry["similar_questions"]:
                                questions.append({
                                    "question": sq["question"],
                                    "dataset": dataset,
                                    "original_question_id": original_question_id,
                                    "similar_question_id": sq.get("id"),
                                    "type": "similar"
                                })
                else:
                    # Regular list format - try to preserve dataset if present
                    for item in data:
                        if isinstance(item, dict):
                            if "question" in item:
                                q_dict = {"question": item["question"]}
                                if "dataset" in item:
                                    q_dict["dataset"] = item["dataset"]
                                questions.append(q_dict)
                            else:
                                questions.append({"question": str(item)})
                        else:
                            questions.append({"question": str(item)})
            elif isinstance(data, dict):
                # Check for similar_questions field (from make_similar_question.py output)
                if "similar_questions" in data:
                    dataset = data.get("dataset")
                    for q in data["similar_questions"]:
                        q_dict = {"question": q["question"]}
                        if dataset:
                            q_dict["dataset"] = dataset
                        questions.append(q_dict)
                    # Also include original question
                    if "original_question" in data:
                        q_dict = {"question": data["original_question"]}
                        if dataset:
                            q_dict["dataset"] = dataset
                        questions.insert(0, q_dict)
                else:
                    questions.append(data)
        except json.JSONDecodeError:
            # If JSON parsing fails, try JSONL format
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        q = json.loads(line)
                        if isinstance(q, dict):
                            if "question" in q:
                                q_dict = {"question": q["question"]}
                                if "dataset" in q:
                                    q_dict["dataset"] = q["dataset"]
                                questions.append(q_dict)
                            else:
                                questions.append({"question": str(q)})
                        elif isinstance(q, str):
                            questions.append({"question": q})
                        else:
                            questions.append({"question": str(q)})
                    except json.JSONDecodeError:
                        questions.append({"question": line.strip()})
    
    return questions

def get_available_datasets():
    """Get list of available math datasets from the eval folder."""
    eval_dir = Path(__file__).parent / "eval" / "chat_benchmarks"
    datasets = []
    
    for item in eval_dir.iterdir():
        if item.is_dir() and (item / "data").exists():
            datasets.append(item.name)
    
    return sorted(datasets)

def load_dataset(dataset_name):
    """Load dataset from the eval folder."""
    data_dir = Path(__file__).parent / "eval" / "chat_benchmarks" / dataset_name / "data"
    
    if not data_dir.exists():
        raise ValueError(f"Dataset '{dataset_name}' not found in eval/chat_benchmarks/")
    
    files = list(data_dir.glob("*.json*"))
    
    questions = []
    
    # Load JSONL files
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    
    return questions

def extract_question_text(question_data):
    """Extract the question text from various dataset formats."""
    if isinstance(question_data, str):
        return question_data
    
    # If it's already a dict with 'question' key (from load_questions_from_file), return it as-is
    if isinstance(question_data, dict) and "question" in question_data:
        return question_data
    
    # Try different field names that might contain the question
    possible_fields = ['problem', 'question', 'prompt', 'text', 'query']
    
    for field in possible_fields:
        if field in question_data:
            return question_data[field]
    
    # If none found, return the first string value
    for value in question_data.values():
        if isinstance(value, str) and len(value) > 20:
            return value
    
    return str(question_data)

def identify_information_to_remove(original_question, model="openai/gpt-4o-mini"):
    """Identify what type of information should be removed consistently across all questions."""
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")
    
    prompt = f"""You are a question editor that identifies key information to remove from math problems.

Given a math problem, identify ONE key piece of information that should be removed. Describe this information in a way that can be consistently applied to similar problems.

For example:
- "the total number of residents/people"
- "the initial quantity"
- "the final result value"
- "the time duration"
- "the distance measurement"

Original Problem:
{original_question}

Output ONLY a short description of what information type should be removed (e.g., "the total number of residents"). Do not include the actual value or explain why, just describe the information type in 5-10 words.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Incomplete Question Generator"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.5,
        "max_tokens": 100
    }
    
    info_type = _openrouter_chat_completion_content(headers=headers, data=data).strip()
    
    return info_type


def generate_incomplete_question_with_guidance(question, info_to_remove, model="openai/gpt-4o-mini"):
    """Generate an incomplete version of a question by removing specific type of information."""
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")
    
    prompt = f"""You are a question editor that creates incomplete versions of math problems for training purposes.

Given a complete math problem, your task is to:

1. Find and replace the following type of information with [BLANK]: {info_to_remove}
2. IMPORTANT: If there is ANY other occurrence or redundant mention of this same information in the problem, you MUST also remove or blank that information
3. The goal is to create a problem where the [BLANK] information truly cannot be inferred from what remains

Rules:
- Replace the specified information with exactly [BLANK]
- Use [BLANK] as the placeholder (not [blank] or other variations)
- Remove or blank any redundant mentions that would allow deducing the blanked value
- Keep the rest of the problem intact and readable
- The incomplete problem should still be understandable, just unsolvable without the missing info

Problem:
{question}

Output ONLY the incomplete version of the problem with [BLANK] replacing the removed information. Do not include explanations, just the modified problem text.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Incomplete Question Generator"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 2000
    }
    
    incomplete_question = _openrouter_chat_completion_content(headers=headers, data=data).strip()
    
    return incomplete_question


def generate_incomplete_questions_with_guidance_batch(
    questions, info_to_remove, model="openai/gpt-4o-mini"
):
    """
    Generate incomplete versions for a list of questions in a single API call.
    Returns a list of incomplete questions aligned with input order.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")
    if not questions:
        return []

    numbered_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = f"""You are a question editor that creates incomplete versions of math problems for training purposes.

Given a list of complete math problems, your task is to:
1. Find and replace the following type of information with [BLANK]: {info_to_remove}
2. IMPORTANT: If there is ANY other occurrence or redundant mention of this same information in the problem, you MUST also remove or blank that information
3. The goal is to create problems where the [BLANK] information truly cannot be inferred from what remains

Rules:
- Replace the specified information with exactly [BLANK]
- Use [BLANK] as the placeholder (not [blank] or other variations)
- Remove or blank any redundant mentions that would allow deducing the blanked value
- Keep the rest of each problem intact and readable
- The incomplete problems should still be understandable, just unsolvable without the missing info

Problems:
{numbered_questions}

Output ONLY a JSON array of strings with the same length and order as the input list.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Incomplete Question Generator (Batch)"
    }

    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 4000
    }

    last = None
    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        content = _openrouter_chat_completion_content(headers=headers, data=data)
        try:
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                incomplete_questions = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON array found", content, 0)
            if not isinstance(incomplete_questions, list):
                raise ValueError("Batch response is not a JSON array")
            return incomplete_questions
        except Exception as e:
            last = e
            if attempt >= OPENROUTER_MAX_RETRIES:
                break
            _sleep_with_backoff(attempt)

    raise ValueError(f"Failed to parse JSON array from batch response after retries: {last}")


def generate_incomplete_questions_by_importance_batch(
    questions: list[str],
    *,
    num_blanks: int = 1,
    model: str = "openai/gpt-4o-mini",
) -> list[str]:
    """
    Generate incomplete questions by blanking out the most important word(s)/short phrase(s).

    The model is instructed to insert EXACTLY `num_blanks` occurrences of the literal token [BLANK]
    (counting occurrences in the final string), selecting the most semantically important
    word/phrase occurrences for solving the problem.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")
    if not questions:
        return []
    num_blanks = int(num_blanks)
    if num_blanks < 1:
        # Degenerate ablation: no blanks -> return originals.
        return list(questions)

    numbered_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    prompt = f"""You are a question editor that creates incomplete versions of math problems.

Given each complete problem, replace the most important word(s) or short phrase(s) (1-3 words)
that are most crucial for solving the problem with the literal token [BLANK].

Hard constraints (must follow):
- Output MUST be a JSON array of strings with the same length and order as the input.
- Each output string MUST contain EXACTLY {num_blanks} occurrences of [BLANK] (count occurrences).
- Prefer blanking NON-NUMERIC keywords/phrases (core objects, conditions, operations, or constraints),
  not generic stopwords. Only blank a number if you truly cannot choose a meaningful keyword.
- Replace specific word/phrase occurrences, not entire sentences.
- Do NOT add or remove other information. Keep the rest of the text unchanged except for the blanks.
- Do NOT output anything except the JSON array.

Problems:
{numbered_questions}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Incomplete Question Generator (Importance Batch)"
    }

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4000,
    }

    last = None
    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        content = _openrouter_chat_completion_content(headers=headers, data=data)
        try:
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                out = json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON array found", content, 0)
            if not isinstance(out, list):
                raise ValueError("Batch response is not a JSON array")
            # Light validation: length + blank count.
            if len(out) != len(questions):
                raise ValueError("Batch response length mismatch")
            for s in out:
                if not isinstance(s, str):
                    raise ValueError("Non-string element in batch response")
                if s.count("[BLANK]") != num_blanks:
                    raise ValueError("Incorrect [BLANK] count in output")
            return out
        except Exception as e:
            last = e
            if attempt >= OPENROUTER_MAX_RETRIES:
                break
            _sleep_with_backoff(attempt)

    raise ValueError(f"Failed to generate/parse importance batch response after retries: {last}")


def generate_incomplete_question_by_importance(
    question: str,
    *,
    num_blanks: int = 1,
    model: str = "openai/gpt-4o-mini",
) -> str:
    """Single-question fallback for importance-based blanking."""
    out = generate_incomplete_questions_by_importance_batch([question], num_blanks=int(num_blanks), model=model)
    return out[0] if out else question


def _parse_json_string_array(content: str, *, expected_len: int) -> list[str]:
    start_idx = content.find("[")
    end_idx = content.rfind("]") + 1
    if start_idx == -1 or end_idx <= start_idx:
        raise ValueError("No JSON array found in model response")
    out = json.loads(content[start_idx:end_idx])
    if not isinstance(out, list) or len(out) != expected_len:
        raise ValueError("Batch response length mismatch")
    for s in out:
        if not isinstance(s, str) or not s.strip():
            raise ValueError("Non-string/empty element in batch response")
    return out


def _perturb_replace_single_number(question: str, *, rng: random.Random) -> str:
    """Replace one numeric literal with a different numeric literal (deterministic)."""
    import re

    matches = list(re.finditer(r"(?<!\w)-?\d+(?:\.\d+)?(?!\w)", question))
    if not matches:
        return question
    preferred = [m for m in matches if m.group() not in ("0", "1", "-1")]
    m = rng.choice(preferred if preferred else matches)
    old = m.group()
    if "." in old:
        val = float(old)
        delta = rng.uniform(0.5, 2.5) * rng.choice([-1.0, 1.0])
        new_val = val + delta
        if abs(new_val - val) < 1e-9:
            new_val = val + 1.0
        new = f"{new_val:.6g}"
    else:
        val = int(old)
        delta = rng.randint(1, max(2, abs(val) // 5 + 1))
        new_val = val + (delta if rng.random() < 0.5 else -delta)
        if new_val == val:
            new_val = val + 1
        new = str(new_val)
    return question[: m.start()] + new + question[m.end() :]


def generate_perturbed_questions_num_replace_batch(
    questions: list[str],
    *,
    model: str = "openai/gpt-4o-mini",
) -> list[str]:
    """Replace exactly one number in each problem with a different number (no [BLANK])."""
    if not questions:
        return []
    deterministic = []
    need_llm: list[tuple[int, str]] = []
    for i, q in enumerate(questions):
        rng = random.Random(hash(q) & 0xFFFFFFFF)
        out = _perturb_replace_single_number(q, rng=rng)
        if out != q:
            deterministic.append((i, out))
        else:
            need_llm.append((i, q))

    results = [None] * len(questions)
    for i, out in deterministic:
        results[i] = out

    if not need_llm:
        return results  # type: ignore[return-value]

    if not OPENROUTER_API_KEY:
        for i, q in need_llm:
            results[i] = q
        return results  # type: ignore[return-value]

    idxs, qs = zip(*need_llm)
    numbered = "\n".join(f"{j + 1}. {q}" for j, q in enumerate(qs))
    prompt = f"""You edit math problems for robustness testing.

For each problem, change EXACTLY ONE numeric literal to a different numeric literal.
- Do NOT use [BLANK].
- Do NOT change anything else (wording, variables, structure).
- The new number must differ from the original.

Problems:
{numbered}

Output ONLY a JSON array of strings with the same length and order as the input."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Num Replace Perturbation (Batch)",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4000,
    }

    last = None
    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        try:
            content = _openrouter_chat_completion_content(headers=headers, data=data)
            llm_out = _parse_json_string_array(content, expected_len=len(qs))
            for pos, out in zip(idxs, llm_out):
                results[pos] = out
            return results  # type: ignore[return-value]
        except Exception as e:
            last = e
            if attempt >= OPENROUTER_MAX_RETRIES:
                break
            _sleep_with_backoff(attempt)

    for i, q in need_llm:
        results[i] = q
    return results  # type: ignore[return-value]


def generate_perturbed_questions_var_rename_batch(
    questions: list[str],
    *,
    model: str = "openai/gpt-4o-mini",
) -> list[str]:
    """Replace one single-letter variable with a random common English word (no [BLANK])."""
    if not questions:
        return []
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")

    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    prompt = f"""You edit math problems for robustness testing.

For each problem, replace EXACTLY ONE single-letter variable (e.g. x, y, n, t) with a random common English noun
(not a single letter). Example: replace "x" with "widget" everywhere that variable appears in the problem, or only
once if the variable appears once — pick one occurrence if multiple variables exist.
- Do NOT use [BLANK].
- Keep all numbers and the problem structure unchanged except for that substitution.

Problems:
{numbered}

Output ONLY a JSON array of strings with the same length and order as the input."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Var Rename Perturbation (Batch)",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 4000,
    }

    last = None
    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        content = _openrouter_chat_completion_content(headers=headers, data=data)
        try:
            return _parse_json_string_array(content, expected_len=len(questions))
        except Exception as e:
            last = e
            if attempt >= OPENROUTER_MAX_RETRIES:
                break
            _sleep_with_backoff(attempt)

    raise ValueError(f"Failed to generate var-rename batch response after retries: {last}")


def generate_perturbed_questions_distractor_batch(
    questions: list[str],
    *,
    model: str = "openai/gpt-4o-mini",
) -> list[str]:
    """Insert one short distractor sentence into each problem (no [BLANK])."""
    if not questions:
        return []
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")

    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    prompt = f"""You edit math problems for robustness testing.

For each problem, insert EXACTLY ONE short distractor sentence (one clause) that is irrelevant or misleading but
plausible-sounding. Place it naturally in the problem text.
- Do NOT use [BLANK].
- Do not change the original numbers or the core question being asked.

Problems:
{numbered}

Output ONLY a JSON array of strings with the same length and order as the input."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Distractor Insert Perturbation (Batch)",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5,
        "max_tokens": 4000,
    }

    last = None
    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        content = _openrouter_chat_completion_content(headers=headers, data=data)
        try:
            return _parse_json_string_array(content, expected_len=len(questions))
        except Exception as e:
            last = e
            if attempt >= OPENROUTER_MAX_RETRIES:
                break
            _sleep_with_backoff(attempt)

    raise ValueError(f"Failed to generate distractor batch response after retries: {last}")


def generate_incomplete_question(original_question, model="openai/gpt-4o-mini"):
    """Generate an incomplete version of a question using OpenRouter API (legacy single-question mode)."""
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")
    
    prompt = f"""You are a question editor that creates incomplete versions of math problems for training purposes.

Given a complete math problem, your task is to:

1. Identify ONE key piece of numerical or specific information that is essential to solve the problem
2. Replace that information with [BLANK] 
3. IMPORTANT: If there is ANY other information in the problem that could be used to deduce or calculate the blanked information, you MUST also remove or blank that information
4. The goal is to create a problem where the [BLANK] information truly cannot be inferred from what remains

Rules:
- Replace ONLY numerical values, specific quantities, or concrete facts (not the problem structure)
- Use exactly [BLANK] as the placeholder
- Remove or blank any redundant information that would allow deducing the blanked value
- Keep the rest of the problem intact and readable
- The incomplete problem should still be understandable, just unsolvable without the missing info

Original Problem:
{original_question}

Output ONLY the incomplete version of the problem with [BLANK] replacing the removed information. Do not include explanations, just the modified problem text.
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Incomplete Question Generator"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    incomplete_question = _openrouter_chat_completion_content(headers=headers, data=data).strip()
    
    return incomplete_question

def main():
    parser = argparse.ArgumentParser(
        description="Generate incomplete versions of questions by removing information and replacing with [BLANK]"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input JSON/JSONL file containing questions (e.g., output from make_similar_question.py)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to use (alternative to --input)"
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=0,
        help="Index of the question to use from the dataset (default: 0, only used with --dataset)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="incomplete_questions.jsonl",
        help="Output file to save the incomplete questions (default: incomplete_questions.jsonl)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="jsonl",
        help="Output format: json or jsonl (default: jsonl)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model to use for generating incomplete questions (default: openai/gpt-4o-mini)"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit"
    )
    parser.add_argument(
        "--consistent",
        action="store_true",
        help="Remove the same type of information consistently across all questions (recommended for similar questions)"
    )
    
    args = parser.parse_args()
    
    # List available datasets if requested
    if args.list_datasets:
        print("\nAvailable datasets:")
        for dataset in get_available_datasets():
            print(f"  - {dataset}")
        return
    
    # Validate input sources
    if not args.input and not args.dataset:
        print("Error: Either --input or --dataset argument is required")
        print("\nUse --input to process a file with questions")
        print("Use --dataset to load from eval/chat_benchmarks/")
        print("Use --list-datasets to see available datasets")
        return
    
    # Load questions
    questions = []
    source_info = ""
    
    if args.input:
        print(f"\nLoading questions from: {args.input}")
        try:
            questions_data = load_questions_from_file(args.input)
            # Extract question text but preserve metadata (dataset, etc.)
            questions = []
            for q in questions_data:
                extracted = extract_question_text(q)
                if isinstance(extracted, dict) and "question" in extracted:
                    # Already in the right format
                    questions.append(extracted)
                else:
                    # Convert to dict format
                    q_dict = {"question": extracted}
                    # Try to preserve dataset if it was in original structure
                    if isinstance(q, dict) and "dataset" in q:
                        q_dict["dataset"] = q["dataset"]
                    questions.append(q_dict)
            source_info = args.input
            print(f"Found {len(questions)} questions in file")
        except Exception as e:
            print(f"Error loading file: {e}")
            return
    else:
        print(f"\nLoading dataset: {args.dataset}")
        try:
            dataset_questions = load_dataset(args.dataset)
            print(f"Found {len(dataset_questions)} questions in dataset")
            
            if args.question_id >= len(dataset_questions):
                print(f"Error: question-id {args.question_id} is out of range (dataset has {len(dataset_questions)} questions)")
                return
            
            question_data = dataset_questions[args.question_id]
            questions = [extract_question_text(question_data)]
            source_info = f"{args.dataset}_q{args.question_id}"
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
    
    if not questions:
        print("No questions found to process!")
        return
    
    print(f"\nProcessing {len(questions)} question(s)...")
    if args.consistent and len(questions) > 1:
        print("🔄 Consistent mode: Will remove the SAME type of information from all questions")
    print("=" * 80)
    
    incomplete_results = []
    info_to_remove = None
    
    # In consistent mode, first identify what to remove from the first question
    if args.consistent and len(questions) > 1:
        print(f"\n[Step 1: Identifying information to remove consistently]")
        first_question_text = questions[0].get("question", questions[0]) if isinstance(questions[0], dict) else questions[0]
        print(f"Analyzing: {first_question_text[:100]}..." if len(first_question_text) > 100 else f"Analyzing: {first_question_text}")
        
        try:
            info_to_remove = identify_information_to_remove(first_question_text, args.model)
            print(f"  ✓ Will remove: '{info_to_remove}' from all questions")
        except Exception as e:
            print(f"  ✗ Error identifying information: {e}")
            print(f"  → Falling back to non-consistent mode")
            args.consistent = False
        
        print("\n" + "=" * 80)
        print(f"\n[Step 2: Applying consistent removal to all {len(questions)} questions]")
    
    # --- Generation (parallel for speed) ---
    prepared = []
    for idx, question_data in enumerate(questions, 1):
        if isinstance(question_data, dict):
            question_text = question_data.get("question", "")
            meta = {
                "dataset": question_data.get("dataset"),
                "original_question_id": question_data.get("original_question_id"),
                "similar_question_id": question_data.get("similar_question_id"),
                "type": question_data.get("type"),
            }
        else:
            question_text = question_data
            meta = {"dataset": None, "original_question_id": None, "similar_question_id": None, "type": None}
        prepared.append((idx, question_text, meta))

    # Fast path: consistent-mode batch generation, with fallback to threaded per-question.
    batch_incomplete = None
    if args.consistent and info_to_remove and len(prepared) > 1:
        try:
            texts = [q for _, q, _ in prepared]
            batch_incomplete = generate_incomplete_questions_with_guidance_batch(
                texts, info_to_remove, args.model
            )
            if len(batch_incomplete) != len(texts):
                batch_incomplete = None
        except Exception:
            batch_incomplete = None

    if batch_incomplete is not None:
        for (idx, question_text, meta), incomplete_question in zip(prepared, batch_incomplete):
            print(f"\n[Question {idx}/{len(prepared)}]")
            if meta.get("dataset"):
                print(f"Dataset: {meta.get('dataset')}")
            print(f"Original: {question_text[:100]}..." if len(question_text) > 100 else f"Original: {question_text}")
            print(f"  ✓ Incomplete: {incomplete_question[:100]}..." if len(incomplete_question) > 100 else f"  ✓ Incomplete: {incomplete_question}")

            result = {
                "id": idx - 1,
                "original_question": question_text,
                "incomplete_question": incomplete_question,
                "source": source_info,
                "has_blank": "[BLANK]" in incomplete_question or "[blank]" in incomplete_question.lower(),
                "info_removed": info_to_remove,
            }
            if meta.get("dataset"):
                result["dataset"] = meta["dataset"]
            if meta.get("original_question_id") is not None:
                result["original_question_id"] = meta["original_question_id"]
            if meta.get("similar_question_id") is not None:
                result["similar_question_id"] = meta["similar_question_id"]
            if meta.get("type"):
                result["type"] = meta["type"]
            incomplete_results.append(result)
    else:
        def _work(question_text: str) -> str:
            if args.consistent and info_to_remove:
                return generate_incomplete_question_with_guidance(
                    question_text, info_to_remove, args.model
                )
            return generate_incomplete_question(question_text, args.model)

        max_workers = max(1, min(OPENROUTER_INCOMPLETE_WORKERS, len(prepared)))
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_work, q): idx for idx, q, _ in prepared}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = e

        for idx, question_text, meta in prepared:
            print(f"\n[Question {idx}/{len(prepared)}]")
            if meta.get("dataset"):
                print(f"Dataset: {meta.get('dataset')}")
            print(f"Original: {question_text[:100]}..." if len(question_text) > 100 else f"Original: {question_text}")

            payload = results.get(idx)
            if isinstance(payload, Exception) or payload is None:
                print(f"  ✗ Error: {payload}")
                result = {
                    "id": idx - 1,
                    "original_question": question_text,
                    "incomplete_question": None,
                    "error": str(payload),
                    "source": source_info,
                }
            else:
                incomplete_question = payload
                print(f"  ✓ Incomplete: {incomplete_question[:100]}..." if len(incomplete_question) > 100 else f"  ✓ Incomplete: {incomplete_question}")
                result = {
                    "id": idx - 1,
                    "original_question": question_text,
                    "incomplete_question": incomplete_question,
                    "source": source_info,
                    "has_blank": "[BLANK]" in incomplete_question or "[blank]" in incomplete_question.lower(),
                }
            if meta.get("dataset"):
                result["dataset"] = meta["dataset"]
            if meta.get("original_question_id") is not None:
                result["original_question_id"] = meta["original_question_id"]
            if meta.get("similar_question_id") is not None:
                result["similar_question_id"] = meta["similar_question_id"]
            if meta.get("type"):
                result["type"] = meta["type"]
            if args.consistent and info_to_remove:
                result["info_removed"] = info_to_remove
            incomplete_results.append(result)
    
    print("\n" + "=" * 80)
    
    # Save results
    if args.format == "jsonl":
        with open(args.output, 'w') as f:
            for result in incomplete_results:
                f.write(json.dumps(result) + '\n')
        print(f"\n✓ Results saved to: {args.output} (JSONL format)")
    else:
        output_data = {
            "source": source_info,
            "model_used": args.model,
            "consistent_mode": args.consistent,
            "total_questions": len(incomplete_results),
            "successful": sum(1 for r in incomplete_results if r.get("incomplete_question")),
            "questions": incomplete_results
        }
        if args.consistent and info_to_remove:
            output_data["info_removed_type"] = info_to_remove
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to: {args.output} (JSON format)")
    
    # Summary
    successful = sum(1 for r in incomplete_results if r.get("incomplete_question"))
    with_blank = sum(1 for r in incomplete_results if r.get("has_blank"))
    
    print(f"\n📊 Summary:")
    print(f"  Total processed: {len(incomplete_results)}")
    print(f"  Successful: {successful}")
    print(f"  With [BLANK]: {with_blank}")
    
    if successful < len(incomplete_results):
        print(f"  Failed: {len(incomplete_results) - successful}")

if __name__ == "__main__":
    main()

