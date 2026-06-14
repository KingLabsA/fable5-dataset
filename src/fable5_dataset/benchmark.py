"""Benchmark generation for evaluating agent performance."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any

from fable5_dataset.preprocessor import Preprocessor


@dataclass
class BenchmarkTask:
    """A single benchmark task."""

    id: str
    category: str
    prompt: str
    expected_tools: list[str] = field(default_factory=list)
    expected_pattern: str = ""
    difficulty: str = "medium"  # "easy", "medium", "hard"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "prompt": self.prompt,
            "expected_tools": self.expected_tools,
            "expected_pattern": self.expected_pattern,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }


_CATEGORIES = {
    "debugging": {
        "description": "Debug and fix code errors",
        "tools": ["read", "edit", "bash"],
        "difficulty": "medium",
    },
    "refactoring": {
        "description": "Refactor code for better structure",
        "tools": ["read", "edit"],
        "difficulty": "medium",
    },
    "implementation": {
        "description": "Implement new features from specifications",
        "tools": ["read", "write", "bash"],
        "difficulty": "hard",
    },
    "exploration": {
        "description": "Explore and understand codebases",
        "tools": ["read", "grep", "glob"],
        "difficulty": "easy",
    },
    "testing": {
        "description": "Write tests for existing code",
        "tools": ["read", "write", "bash"],
        "difficulty": "medium",
    },
    "documentation": {
        "description": "Write documentation for code",
        "tools": ["read", "write"],
        "difficulty": "easy",
    },
    "security": {
        "description": "Identify and fix security issues",
        "tools": ["read", "edit", "bash"],
        "difficulty": "hard",
    },
    "optimization": {
        "description": "Optimize code performance",
        "tools": ["read", "edit", "bash"],
        "difficulty": "hard",
    },
}

_PROMPT_TEMPLATES = {
    "debugging": [
        "Fix the bug in {module} where {symptom}",
        "The function {function} in {module} is returning incorrect values. Debug and fix it.",
        "There's a regression in {module} causing {symptom}. Find and fix the issue.",
        "The test suite for {module} is failing with error: {symptom}. Fix the code.",
        "Debug why {module} crashes when {condition}",
    ],
    "refactoring": [
        "Refactor {module} to use the strategy pattern instead of if-else chains",
        "Extract the duplicated logic in {module} into a shared utility function",
        "Simplify the class hierarchy in {module} by removing unnecessary abstractions",
        "Convert the callback-based API in {module} to use async/await",
        "Reorganize {module} to separate concerns between data access and business logic",
    ],
    "implementation": [
        "Implement a caching layer for {module} that invalidates on {condition}",
        "Add a REST API endpoint for {resource} with CRUD operations and validation",
        "Create a data pipeline that processes {input} and produces {output}",
        "Implement rate limiting for the {module} service with configurable thresholds",
        "Build a command-line interface for {module} with subcommands for {operations}",
    ],
    "exploration": [
        "Find all places in the codebase where {pattern} is used",
        "Explain the architecture of the {module} system",
        "Identify all dependencies of {module} and their purposes",
        "Map out the data flow from {entry} through {module}",
        "What design patterns are used in {module}?",
    ],
    "testing": [
        "Write unit tests for {function} in {module} covering edge cases",
        "Create integration tests for the {module} API endpoints",
        "Add property-based tests for the {module} parser",
        "Write regression tests for the bug fixed in {commit}",
        "Create test fixtures and factories for {module} models",
    ],
    "documentation": [
        "Write API documentation for {module} including all public methods",
        "Add inline comments explaining the complex logic in {function}",
        "Create a README for {module} with setup, usage, and examples",
        "Document the configuration options for {module}",
        "Write a migration guide from v1 to v2 of {module}",
    ],
    "security": [
        "Identify and fix the SQL injection vulnerability in {module}",
        "Audit {module} for security issues and provide a report with fixes",
        "Fix the authentication bypass in {module} and add proper validation",
        "Implement Content Security Policy headers for {module}",
        "Add input sanitization to all user-facing endpoints in {module}",
    ],
    "optimization": [
        "Optimize the database queries in {module} to reduce latency by 50%",
        "Profile and optimize the hot path in {function} for better throughput",
        "Reduce memory usage in {module} by implementing lazy loading",
        "Implement connection pooling for {module} to handle higher load",
        "Optimize the data processing pipeline in {module} for batch operations",
    ],
}


class BenchmarkGenerator:
    """Generate benchmark tasks from Fable5 datasets."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.preprocessor = Preprocessor()

    def generate_benchmark(
        self,
        records: list[dict[str, Any]],
        num_tasks: int = 50,
        categories: list[str] | None = None,
    ) -> list[BenchmarkTask]:
        """Generate benchmark tasks from records.

        Args:
            records: List of normalized records.
            num_tasks: Number of tasks to generate.
            categories: Optional list of categories to include.

        Returns:
            List of BenchmarkTask objects.
        """
        if categories is None:
            categories = list(_CATEGORIES.keys())

        tasks: list[BenchmarkTask] = []
        available_records = list(records)
        self.rng.shuffle(available_records)

        tasks_per_category = max(1, num_tasks // len(categories))

        for category in categories:
            if category not in _CATEGORIES:
                continue

            cat_config = _CATEGORIES[category]
            templates = _PROMPT_TEMPLATES.get(category, [])

            for i in range(tasks_per_category):
                if len(tasks) >= num_tasks:
                    break

                record_idx = (i * 7 + category.__hash__()) % max(len(available_records), 1)
                record = available_records[record_idx % len(available_records)] if available_records else {}

                template = templates[i % len(templates)] if templates else f"Complete a {category} task"

                prompt = self._fill_template(template, record, category)

                task_id = hashlib.sha256(f"{category}_{i}".encode()).hexdigest()[:12]
                task = BenchmarkTask(
                    id=f"fable5_{category}_{task_id}",
                    category=category,
                    prompt=prompt,
                    expected_tools=cat_config["tools"],
                    expected_pattern=category,
                    difficulty=cat_config["difficulty"],
                    metadata={
                        "source_record": record.get("id", ""),
                        "template_idx": i,
                    },
                )
                tasks.append(task)

        return tasks[:num_tasks]

    def generate_from_records(
        self,
        records: list[dict[str, Any]],
        num_tasks: int = 50,
    ) -> list[BenchmarkTask]:
        """Generate benchmark tasks directly from record content.

        Creates tasks that mirror real agent interactions from the dataset.
        """
        tasks: list[BenchmarkTask] = []
        self.rng.shuffle(records)

        for i, record in enumerate(records):
            if len(tasks) >= num_tasks:
                break

            messages = record.get("messages", [])
            if not messages:
                continue

            user_messages = [m for m in messages if m.get("role") == "user"]
            if not user_messages:
                continue

            prompt = user_messages[0].get("content", "")
            if not prompt or len(prompt) < 10:
                continue

            tools_used = set()
            for tool in record.get("tools", []):
                tools_used.add(tool.get("name", ""))

            category = self._infer_category(prompt, tools_used)
            difficulty = self._infer_difficulty(len(messages), len(tools_used))

            task_id = hashlib.sha256(f"record_{i}".encode()).hexdigest()[:12]
            task = BenchmarkTask(
                id=f"fable5_real_{task_id}",
                category=category,
                prompt=prompt,
                expected_tools=list(tools_used) if tools_used else _CATEGORIES.get(category, {}).get("tools", []),
                expected_pattern=category,
                difficulty=difficulty,
                metadata={"source_record": record.get("id", "")},
            )
            tasks.append(task)

        return tasks

    def _fill_template(self, template: str, record: dict, category: str) -> str:
        """Fill a prompt template with context from a record."""
        module_names = ["auth", "users", "orders", "payments", "notifications", "reports", "analytics", "cache"]
        function_names = ["authenticate", "process_order", "calculate_total", "validate_input", "fetch_data"]
        symptoms = ["null pointer exceptions", "incorrect output", "timeout errors", "memory leaks", "race conditions"]
        conditions = ["concurrent access", "large input sizes", "network failures", "empty data"]
        resources = ["users", "orders", "products", "sessions", "reports"]
        operations = ["create, read, update, delete", "list, search, filter", "import, export, sync"]

        module = self.rng.choice(module_names)
        function = self.rng.choice(function_names)
        symptom = self.rng.choice(symptoms)
        condition = self.rng.choice(conditions)
        resource = self.rng.choice(resources)
        operations_str = self.rng.choice(operations)

        try:
            return template.format(
                module=module, function=function, symptom=symptom,
                condition=condition, resource=resource, operations=operations_str,
                pattern=category, input=module, output=f"processed_{module}",
                entry=f"{module}_handler", commit=f"abc{self.rng.randint(100, 999)}",
            )
        except KeyError:
            return template

    def _infer_category(self, prompt: str, tools: set[str]) -> str:
        prompt_lower = prompt.lower()

        if any(w in prompt_lower for w in ["bug", "fix", "debug", "error", "crash", "failing"]):
            return "debugging"
        if any(w in prompt_lower for w in ["refactor", "restructure", "clean", "simplify", "reorganize"]):
            return "refactoring"
        if any(w in prompt_lower for w in ["implement", "add", "create", "build", "new feature"]):
            return "implementation"
        if any(w in prompt_lower for w in ["find", "search", "where", "explain", "understand", "explore"]):
            return "exploration"
        if any(w in prompt_lower for w in ["test", "spec", "coverage", "unit test"]):
            return "testing"
        if any(w in prompt_lower for w in ["document", "docs", "readme", "comment", "explain"]):
            return "documentation"
        if any(w in prompt_lower for w in ["security", "vulnerability", "inject", "auth", "sanitize"]):
            return "security"
        if any(w in prompt_lower for w in ["optimize", "performance", "speed", "memory", "cache"]):
            return "optimization"

        if "bash" in tools and ("edit" in tools or "write" in tools):
            return "debugging"
        if "read" in tools and "grep" in tools and "edit" not in tools and "write" not in tools:
            return "exploration"

        return "implementation"

    def _infer_difficulty(self, num_messages: int, num_tools: int) -> str:
        if num_messages > 15 or num_tools > 4:
            return "hard"
        elif num_messages > 6 or num_tools > 2:
            return "medium"
        return "easy"

    def save_benchmark(self, tasks: list[BenchmarkTask], path: str) -> None:
        """Save benchmark tasks to a JSONL file."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for task in tasks:
                f.write(json.dumps(task.to_dict()) + "\n")

    def load_benchmark(self, path: str) -> list[BenchmarkTask]:
        """Load benchmark tasks from a JSONL file."""
        tasks = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    tasks.append(BenchmarkTask(
                        id=data["id"],
                        category=data["category"],
                        prompt=data["prompt"],
                        expected_tools=data.get("expected_tools", []),
                        expected_pattern=data.get("expected_pattern", ""),
                        difficulty=data.get("difficulty", "medium"),
                        metadata=data.get("metadata", {}),
                    ))
        return tasks
