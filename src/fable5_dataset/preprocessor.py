"""Preprocessing for Fable5 datasets."""

from __future__ import annotations

import re
from typing import Any


class Preprocessor:
    """Preprocess Fable5 dataset records."""

    _PII_PATTERNS = [
        (re.compile(r'/Users/\w+/'), '/Users/[REDACTED]/'),
        (re.compile(r'/home/\w+/'), '/home/[REDACTED]/'),
        (re.compile(r'/home/\w+/'), '/home/[REDACTED]/'),
        (re.compile(r'C:\\Users\\\w+\\'), 'C:\\Users\\[REDACTED]\\'),
        (re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b'), '[EMAIL_REDACTED]'),
        (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), '[PHONE_REDACTED]'),
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN_REDACTED]'),
        (re.compile(r'\b(?:ssh|https?|ftp)://\S+@'), '[URL_CREDENTIALS_REDACTED]@'),
        (re.compile(r'\b(?:password|passwd|secret|token|api_key|apikey)\s*[:=]\s*\S+', re.IGNORECASE), '[CREDENTIALS_REDACTED]'),
    ]

    def normalize_format(self, records: list[dict[str, Any]], source_format: str = "glint") -> list[dict[str, Any]]:
        """Normalize all records to a unified schema.

        Unified schema:
        {
            "id": str,
            "messages": [{"role": str, "content": str}],
            "tools": [{"name": str, "input": dict}],
            "metadata": dict
        }
        """
        normalized = []
        for record in records:
            try:
                if source_format == "glint":
                    norm = self._normalize_glint(record)
                elif source_format == "armand0e":
                    norm = self._normalize_armand0e(record)
                elif source_format == "vfable":
                    norm = self._normalize_vfable(record)
                elif source_format == "opencoven":
                    norm = self._normalize_opencoven(record)
                elif source_format == "victor":
                    norm = self._normalize_victor(record)
                elif source_format == "coding_excellence":
                    norm = self._normalize_coding_excellence(record)
                else:
                    norm = self._auto_normalize(record)
                normalized.append(norm)
            except Exception:
                normalized.append(self._auto_normalize(record))

        return normalized

    def remove_pii(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove PII from all records.

        Removes: user paths, emails, phone numbers, SSNs, credentials.
        """
        cleaned = []
        for record in records:
            cleaned_record = self._remove_pii_from_record(record)
            cleaned.append(cleaned_record)
        return cleaned

    def filter_quality(self, records: list[dict[str, Any]], min_quality: float = 0.5) -> list[dict[str, Any]]:
        """Filter records by quality score.

        Quality is computed based on: reasoning length, tool diversity,
        error recovery rate, and response completeness.
        """
        filtered = []
        for record in records:
            quality = self._compute_quality(record)
            if quality >= min_quality:
                record["_quality_score"] = quality
                filtered.append(record)
        return filtered

    def _normalize_glint(self, record: dict) -> dict[str, Any]:
        messages = []
        tools = []
        for turn in record.get("turns", []):
            msg = {"role": turn.get("role", "user"), "content": turn.get("content", "")}
            content = turn.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tools.append({"name": block.get("name", ""), "input": block.get("input", {})})
                msg["content"] = "\n".join(text_parts)
            messages.append(msg)

        return {
            "id": record.get("session_id", ""),
            "messages": messages,
            "tools": tools,
            "metadata": record.get("metadata", {}),
        }

    def _normalize_armand0e(self, record: dict) -> dict[str, Any]:
        messages = []
        tools = []
        for turn in record.get("conversation", []):
            msg = {"role": turn.get("role", "user"), "content": turn.get("content", "")}
            tool_calls = turn.get("tool_calls", [])
            for tc in tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    tools.append({"name": func.get("name", ""), "input": func.get("arguments", {})})
            messages.append(msg)

        return {
            "id": record.get("id", ""),
            "messages": messages,
            "tools": tools,
            "metadata": record.get("metadata", {}),
        }

    def _normalize_vfable(self, record: dict) -> dict[str, Any]:
        messages = []
        tools = []
        for turn in record.get("trajectory", []):
            msg = {"role": turn.get("role", "user"), "content": turn.get("content", "")}
            tool_use = turn.get("tool_use")
            if tool_use and isinstance(tool_use, dict):
                tools.append({"name": tool_use.get("name", ""), "input": tool_use.get("input", {})})
            messages.append(msg)

        return {
            "id": record.get("id", ""),
            "messages": messages,
            "tools": tools,
            "metadata": record.get("metadata", {}),
        }

    def _normalize_opencoven(self, record: dict) -> dict[str, Any]:
        messages = [
            {"role": "user", "content": record.get("source", "")},
            {"role": "assistant", "content": record.get("target", "")},
        ]

        return {
            "id": record.get("id", ""),
            "messages": messages,
            "tools": [],
            "metadata": {k: v for k, v in record.items() if k not in ("source", "target", "id")},
        }

    def _normalize_victor(self, record: dict) -> dict[str, Any]:
        response = record.get("response", "")
        messages = [
            {"role": "user", "content": record.get("prompt", "")},
            {"role": "assistant", "content": response if isinstance(response, str) else str(response)},
        ]

        return {
            "id": record.get("id", ""),
            "messages": messages,
            "tools": [],
            "metadata": {k: v for k, v in record.items() if k not in ("prompt", "response", "id")},
        }

    def _normalize_coding_excellence(self, record: dict) -> dict[str, Any]:
        messages = []
        tools = []
        for turn in record.get("turns", []):
            msg = {"role": turn.get("role", "user"), "content": turn.get("content", "")}
            content = turn.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tools.append({"name": block.get("name", ""), "input": block.get("input", {})})
                msg["content"] = "\n".join(text_parts)
            messages.append(msg)

        metadata = record.get("metadata", {})
        metadata["quality_score"] = record.get("quality_score", 0.0)

        return {
            "id": record.get("session_id", ""),
            "messages": messages,
            "tools": tools,
            "metadata": metadata,
        }

    def _auto_normalize(self, record: dict) -> dict[str, Any]:
        messages = []
        tools = []

        if "messages" in record:
            for msg in record["messages"]:
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        elif "turns" in record:
            for turn in record["turns"]:
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        elif "conversation" in record:
            for turn in record["conversation"]:
                messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        elif "source" in record and "target" in record:
            messages = [
                {"role": "user", "content": record["source"]},
                {"role": "assistant", "content": record["target"]},
            ]
        elif "prompt" in record and "response" in record:
            messages = [
                {"role": "user", "content": record["prompt"]},
                {"role": "assistant", "content": str(record["response"])},
            ]

        return {
            "id": record.get("id", record.get("session_id", "")),
            "messages": messages,
            "tools": tools,
            "metadata": record.get("metadata", {}),
        }

    def _remove_pii_from_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Remove PII from a single record."""
        cleaned = {}
        for key, value in record.items():
            if isinstance(value, str):
                cleaned[key] = self._remove_pii_from_string(value)
            elif isinstance(value, list):
                cleaned[key] = [self._remove_pii_from_item(item) for item in value]
            elif isinstance(value, dict):
                cleaned[key] = self._remove_pii_from_record(value)
            else:
                cleaned[key] = value
        return cleaned

    def _remove_pii_from_string(self, text: str) -> str:
        for pattern, replacement in self._PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def _remove_pii_from_item(self, item: Any) -> Any:
        if isinstance(item, str):
            return self._remove_pii_from_string(item)
        elif isinstance(item, dict):
            return self._remove_pii_from_record(item)
        elif isinstance(item, list):
            return [self._remove_pii_from_item(i) for i in item]
        return item

    def _compute_quality(self, record: dict[str, Any]) -> float:
        """Compute a quality score for a record."""
        messages = record.get("messages", [])
        if not messages:
            metadata_quality = record.get("metadata", {}).get("quality_score", 0.0)
            if isinstance(metadata_quality, (int, float)):
                return float(metadata_quality)
            return 0.0

        quality = record.get("quality_score", 0.0)
        if isinstance(quality, (int, float)) and quality > 0:
            return float(quality)

        total_chars = sum(len(m.get("content", "")) for m in messages)
        length_score = min(total_chars / 3000.0, 1.0) * 0.3

        unique_tools = set()
        for tool in record.get("tools", []):
            unique_tools.add(tool.get("name", ""))
        diversity_score = min(len(unique_tools) / 3.0, 1.0) * 0.3 if unique_tools else 0.1

        has_assistant = any(m.get("role") == "assistant" for m in messages)
        has_user = any(m.get("role") == "user" for m in messages)
        completeness = (0.5 if has_assistant else 0.0) + (0.5 if has_user else 0.0)
        completeness_score = completeness * 0.4

        return length_score + diversity_score + completeness_score
