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
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")

OPENROUTER_MAX_RETRIES = int(os.getenv("OPENROUTER_MAX_RETRIES", "6"))
OPENROUTER_RETRY_BASE_SECONDS = float(os.getenv("OPENROUTER_RETRY_BASE_SECONDS", "1.0"))
OPENROUTER_RETRY_MAX_SECONDS = float(os.getenv("OPENROUTER_RETRY_MAX_SECONDS", "30.0"))
OPENROUTER_TIMEOUT_SECONDS = float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "90"))
OPENROUTER_SIMILAR_WORKERS = int(os.getenv("OPENROUTER_SIMILAR_WORKERS", "6"))
OPENROUTER_QUIET = os.getenv("OPENROUTER_QUIET", "0") == "1"


def _sleep_with_backoff(attempt: int) -> None:
    base = min(OPENROUTER_RETRY_MAX_SECONDS, OPENROUTER_RETRY_BASE_SECONDS * (2 ** attempt))
    jitter = 0.5 + random.random()  # [0.5, 1.5)
    time.sleep(base * jitter)


def _openrouter_chat_completion_content(*, headers: dict, data: dict) -> str:
    last_err: Exception | None = None
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


def _parse_json_array_fallback(content: str) -> list[str]:
    # Parse the JSON array from the response (best-effort).
    try:
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            arr = json.loads(json_str)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
    except Exception:
        pass
    # Fallback: split by newlines
    return [q.strip() for q in content.split('\n') if q.strip() and len(q.strip()) > 20]


def get_available_datasets():
    """Get list of available math datasets from the eval folder."""
    eval_dir = Path(__file__).parent / "eval" / "chat_benchmarks"
    datasets = []
    
    for item in eval_dir.iterdir():
        if item.is_dir() and (item / "data").exists():
            datasets.append(item.name)
    
    return sorted(datasets)

def load_dataset(dataset_name):
    """Load dataset from /data/dataset_name/test.json or train.json."""
    data_dir = Path(f"../data/{dataset_name}")
    
    if not data_dir.exists():
        raise ValueError(f"Dataset directory '{data_dir}' not found")
    
    questions = []
    
    # Try test.json first, then train.json
    file_path = data_dir / "test.json"
    if not file_path.exists():
        file_path = data_dir / "train.json"
    
    if not file_path.exists():
        raise ValueError(f"Neither test.json nor train.json found in {data_dir}")
    
    # Load the file (handle both JSON and JSONL formats)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
        # First, try to parse as regular JSON (could be pretty-printed)
        try:
            data = json.loads(content)
            if isinstance(data, list):
                questions = data
            else:
                questions = [data]
        except json.JSONDecodeError:
            # If JSON parsing fails, try JSONL format (one JSON object per line)
            questions = []
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    try:
                        questions.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
    
    return questions

def extract_question_text(question_data):
    """Extract the question text from various dataset formats."""
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

def generate_similar_questions(original_question, num_questions=5, model="openai/gpt-4o-mini"):
    """Generate similar questions using OpenRouter API."""
    
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in .env file")
    
    prompt = f"""You are a math problem generator. Given an original math problem, create {num_questions} similar problems that:

1. Follow the EXACT same structure and solution method as the original
2. Use DIFFERENT numerical values (change all numbers to make the problem unique)
3. Maintain the same difficulty level
4. Have the same type of solution approach
5. Are valid, solvable problems

Original Problem:
{original_question}

Generate {num_questions} similar problems. For each problem:
- Change ALL numerical values to create unique scenarios
- Keep the problem structure and mathematical concepts identical
- Ensure the problem remains solvable and realistic
- Make sure the new numbers create valid mathematical relationships

Output ONLY a JSON array of {num_questions} similar problems, where each element is a string containing the full problem text. Do not include solutions or explanations, only the problems.

Format your response as:
["Problem 1 text here...", "Problem 2 text here...", ...]
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/contamination-train",
        "X-Title": "Similar Question Generator"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.8,
        "max_tokens": 4000
    }

    # Try multiple times if we can't parse enough outputs (or response is truncated).
    last = None
    for attempt in range(OPENROUTER_MAX_RETRIES + 1):
        if not OPENROUTER_QUIET:
            print("\nCalling OpenRouter API with gpt-4o-mini...")
        content = _openrouter_chat_completion_content(headers=headers, data=data)
        similar_questions = _parse_json_array_fallback(content)
        if len(similar_questions) >= max(1, num_questions):
            return similar_questions[:num_questions]
        last = similar_questions
        if attempt >= OPENROUTER_MAX_RETRIES:
            break
        _sleep_with_backoff(attempt)

    # Best effort return.
    return (last or [])[:num_questions]

def main():
    parser = argparse.ArgumentParser(
        description="Generate similar math questions with different numerical setups using OpenRouter API"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of the dataset to use (e.g., AIME24, MATH500, AMC23)"
    )
    parser.add_argument(
        "--question-id",
        type=int,
        default=1,
        help="End index for the slice of questions to process (e.g., --question-id 1 processes questions 0 to 1, default: 1)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=5,
        help="Number of similar questions to generate (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="similar_questions.json",
        help="Output file to save the generated questions (optional, will print to console if not specified)"
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/o4-mini",
        help="Model to use for generating similar questions (default: openai/gpt-4o-mini)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="Output format: json (single file) or jsonl (one question per line)"
    )
    
    args = parser.parse_args()
    
    # List available datasets if requested
    if args.list_datasets:
        print("\nAvailable datasets:")
        for dataset in get_available_datasets():
            print(f"  - {dataset}")
        return
    
    # Validate dataset argument
    if not args.dataset:
        print("Error: --dataset argument is required")
        print("\nUse --list-datasets to see available datasets")
        print("\nAvailable datasets:")
        for dataset in get_available_datasets():
            print(f"  - {dataset}")
        return
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    try:
        questions = load_dataset(args.dataset)
        print(f"Found {len(questions)} questions in dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Validate question ID (end index for slice)
    if args.question_id < 1:
        print(f"Error: question-id must be at least 1 (got {args.question_id})")
        return
    if args.question_id > len(questions):
        print(f"Error: question-id {args.question_id} is out of range (dataset has {len(questions)} questions, max end index: {len(questions)})")
        return
    
    # Get slice of questions to process (0 to question_id)
    questions_to_process = questions[0:args.question_id]
    print(f"\nProcessing questions 0 to {args.question_id-1} ({len(questions_to_process)} question(s))")
    
    all_output_data = []
    
    # Process each question in the slice
    # Parallelize OpenRouter calls across questions for speed.
    def _work(q_idx: int, q_data):
        original_question = extract_question_text(q_data)
        similars = generate_similar_questions(original_question, args.num_questions, args.model)
        return q_idx, original_question, similars

    results_by_idx = {}
    max_workers = max(1, min(OPENROUTER_SIMILAR_WORKERS, len(questions_to_process)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_work, i, qd): i for i, qd in enumerate(questions_to_process)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                q_idx, original_question, similar_questions = fut.result()
                results_by_idx[q_idx] = (original_question, similar_questions)
            except Exception as e:
                results_by_idx[i] = (None, e)

    # Print/save in original order for readability.
    for question_idx in range(len(questions_to_process)):
        entry = results_by_idx.get(question_idx)
        if entry is None:
            continue
        original_question, payload = entry
        if original_question is None or isinstance(payload, Exception):
            print(f"\n✗ Error generating questions for question {question_idx}: {payload}")
            continue
        similar_questions = payload

        print(f"\n{'='*80}")
        print(f"Original Question (ID: {question_idx}):")
        print("-" * 80)
        print(original_question)
        print("-" * 80)
        print(f"\n✓ Generated {len(similar_questions)} similar questions:")
        print("=" * 80)

        output_data = {
            "dataset": args.dataset,
            "original_question_id": question_idx,
            "original_question": original_question,
            "similar_questions": []
        }
        for i, question in enumerate(similar_questions, 1):
            print(f"\n[Similar Question {i}]")
            print(question)
            print()
            output_data["similar_questions"].append({"id": i, "question": question})
        all_output_data.append(output_data)
        print(f"\n✓ Successfully processed question {question_idx}")
    
    # Save to file if output path specified
    if args.output and all_output_data:
        if args.format == "jsonl":
            # Save as JSONL (one question per line)
            with open(args.output, 'w') as f:
                for output_data in all_output_data:
                    question_idx = output_data["original_question_id"]
                    original_question = output_data["original_question"]
                    
                    # Write original question
                    f.write(json.dumps({
                        "id": f"{args.dataset}_original_{question_idx}",
                        "question": original_question,
                        "source": args.dataset,
                        "type": "original"
                    }) + '\n')
                    
                    # Write similar questions
                    for similar_q in output_data["similar_questions"]:
                        f.write(json.dumps({
                            "id": f"{args.dataset}_similar_{question_idx}_{similar_q['id']}",
                            "question": similar_q["question"],
                            "source": args.dataset,
                            "type": "similar",
                            "original_id": question_idx
                        }) + '\n')
            print(f"\n✓ Results saved to: {args.output} (JSONL format)")
        else:
            # Save as JSON (default)
            # If only one question, save in original format; otherwise save as array
            if len(all_output_data) == 1:
                with open(args.output, 'w') as f:
                    json.dump(all_output_data[0], f, indent=2)
            else:
                with open(args.output, 'w') as f:
                    json.dump(all_output_data, f, indent=2)
            print(f"\n✓ Results saved to: {args.output} (JSON format)")
        
        print("=" * 80)
        print(f"\n✓ Successfully processed {len(all_output_data)} question(s)!")

if __name__ == "__main__":
    main()
