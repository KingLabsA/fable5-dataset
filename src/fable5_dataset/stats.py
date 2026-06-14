"""Dataset statistics computation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetStatistics:
    """Statistics about a dataset."""

    total_rows: int = 0
    unique_sessions: int = 0
    total_messages: int = 0
    total_tokens_estimated: int = 0
    avg_turns_per_session: float = 0.0
    avg_message_length: float = 0.0
    tool_distribution: dict[str, int] = field(default_factory=dict)
    role_distribution: dict[str, int] = field(default_factory=dict)
    category_distribution: dict[str, int] = field(default_factory=dict)
    quality_score_avg: float = 0.0
    error_rate: float = 0.0
    min_turns: int = 0
    max_turns: int = 0
    median_turns: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "unique_sessions": self.unique_sessions,
            "total_messages": self.total_messages,
            "total_tokens_estimated": self.total_tokens_estimated,
            "avg_turns_per_session": round(self.avg_turns_per_session, 2),
            "avg_message_length": round(self.avg_message_length, 2),
            "tool_distribution": dict(sorted(self.tool_distribution.items(), key=lambda x: -x[1])),
            "role_distribution": dict(sorted(self.role_distribution.items(), key=lambda x: -x[1])),
            "category_distribution": self.category_distribution,
            "quality_score_avg": round(self.quality_score_avg, 3),
            "error_rate": round(self.error_rate, 4),
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "median_turns": round(self.median_turns, 2),
        }

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "DATASET STATISTICS",
            "=" * 50,
            f"Total records:       {self.total_rows:,}",
            f"Unique sessions:     {self.unique_sessions:,}",
            f"Total messages:      {self.total_messages:,}",
            f"Est. total tokens:   {self.total_tokens_estimated:,}",
            "",
            f"Avg turns/session:   {self.avg_turns_per_session:.1f}",
            f"Avg message length:  {self.avg_message_length:.0f} chars",
            f"Min turns:           {self.min_turns}",
            f"Max turns:           {self.max_turns}",
            f"Median turns:        {self.median_turns:.1f}",
            "",
            f"Quality score avg:   {self.quality_score_avg:.3f}",
            f"Error rate:          {self.error_rate:.2%}",
            "",
            "Tool Distribution:",
        ]
        for tool, count in sorted(self.tool_distribution.items(), key=lambda x: -x[1])[:10]:
            pct = count / max(self.total_messages, 1) * 100
            lines.append(f"  {tool:15s} {count:6d} ({pct:5.1f}%)")

        lines.extend(["", "Role Distribution:"])
        for role, count in sorted(self.role_distribution.items(), key=lambda x: -x[1]):
            pct = count / max(self.total_messages, 1) * 100
            lines.append(f"  {role:15s} {count:6d} ({pct:5.1f}%)")

        lines.append("=" * 50)
        return "\n".join(lines)


class DatasetStats:
    """Compute statistics on Fable5 datasets."""

    def compute_stats(self, records: list[dict[str, Any]]) -> DatasetStatistics:
        """Compute comprehensive statistics for a dataset.

        Args:
            records: List of normalized records.

        Returns:
            DatasetStatistics with computed metrics.
        """
        stats = DatasetStatistics()
        stats.total_rows = len(records)

        if not records:
            return stats

        session_ids: set[str] = set()
        total_messages = 0
        total_chars = 0
        tool_counter: Counter = Counter()
        role_counter: Counter = Counter()
        quality_scores = []
        error_count = 0
        turn_counts = []

        for record in records:
            sid = record.get("id", record.get("session_id", ""))
            if sid:
                session_ids.add(sid)

            messages = record.get("messages", [])
            num_turns = len(messages)
            turn_counts.append(num_turns)
            total_messages += num_turns

            for msg in messages:
                content = msg.get("content", "")
                total_chars += len(content)
                role_counter[msg.get("role", "unknown")] += 1

                content_lower = content.lower()
                if any(w in content_lower for w in ["error", "exception", "failed", "traceback"]):
                    error_count += 1

            for tool in record.get("tools", []):
                name = tool.get("name", "unknown")
                tool_counter[name] += 1

            quality = record.get("quality_score", record.get("metadata", {}).get("quality_score", 0.0))
            if isinstance(quality, (int, float)) and quality > 0:
                quality_scores.append(float(quality))

        stats.unique_sessions = len(session_ids) if session_ids else stats.total_rows
        stats.total_messages = total_messages
        stats.total_tokens_estimated = total_chars // 4
        stats.avg_turns_per_session = total_messages / max(stats.total_rows, 1)
        stats.avg_message_length = total_chars / max(total_messages, 1)
        stats.tool_distribution = dict(tool_counter)
        stats.role_distribution = dict(role_counter)
        stats.quality_score_avg = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        stats.error_rate = error_count / max(total_messages, 1)

        if turn_counts:
            sorted_turns = sorted(turn_counts)
            stats.min_turns = min(turn_counts)
            stats.max_turns = max(turn_counts)
            mid = len(sorted_turns) // 2
            stats.median_turns = float(sorted_turns[mid])

        stats.category_distribution = self._compute_category_distribution(records)

        return stats

    def _compute_category_distribution(self, records: list[dict[str, Any]]) -> dict[str, int]:
        categories: Counter = Counter()
        for record in records:
            messages = record.get("messages", [])
            category = self._infer_category_from_messages(messages)
            categories[category] += 1
        return dict(categories)

    def _infer_category_from_messages(self, messages: list[dict]) -> str:
        if not messages:
            return "unknown"

        all_content = " ".join(m.get("content", "").lower() for m in messages)
        tools = set()

        category_keywords = {
            "debugging": ["bug", "fix", "debug", "error", "crash", "failing", "traceback", "exception"],
            "implementation": ["implement", "add", "create", "build", "new feature", "write", "develop"],
            "exploration": ["find", "search", "where", "understand", "explore", "explain", "what does"],
            "refactoring": ["refactor", "restructure", "clean", "simplify", "reorganize", "improve"],
            "testing": ["test", "spec", "coverage", "unit test", "integration test"],
            "documentation": ["document", "docs", "readme", "comment", "guide", "tutorial"],
        }

        best_category = "other"
        best_count = 0
        for category, keywords in category_keywords.items():
            count = sum(1 for kw in keywords if kw in all_content)
            if count > best_count:
                best_count = count
                best_category = category

        return best_category

    def compute_stats_from_file(self, path: str | Path) -> DatasetStatistics:
        """Compute statistics from a JSONL file."""
        path = Path(path)
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return self.compute_stats(records)

    def compare_datasets(self, datasets: dict[str, list[dict[str, Any]]]) -> dict[str, DatasetStatistics]:
        """Compare statistics across multiple datasets.

        Args:
            datasets: Dict mapping dataset names to record lists.

        Returns:
            Dict mapping dataset names to their DatasetStatistics.
        """
        results = {}
        for name, records in datasets.items():
            results[name] = self.compute_stats(records)
        return results
