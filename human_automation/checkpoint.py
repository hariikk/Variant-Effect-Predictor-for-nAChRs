"""
Checkpoint system for resuming interrupted scraping runs.
Tracks which PMIDs have been processed, their status, and results.
"""

import json
from datetime import datetime
from pathlib import Path

from config import CHECKPOINT_FILE


class CheckpointManager:
    """Manages checkpoint state for the scraping pipeline."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_FILE):
        self.path = checkpoint_path
        self.data = self._load()

    def _load(self) -> dict:
        """Load checkpoint from disk, or return empty state."""
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"papers": {}, "search_results": {}}

    def save(self):
        """Persist checkpoint to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)

    def is_processed(self, pmid: str) -> bool:
        """Check if a PMID has already been processed (successfully or marked inaccessible)."""
        status = self.data["papers"].get(str(pmid), {}).get("status")
        return status in ("processed", "inaccessible")

    def mark_processed(self, pmid: str, entries_found: int):
        """Mark a PMID as successfully processed."""
        self.data["papers"][str(pmid)] = {
            "status": "processed",
            "entries_found": entries_found,
            "timestamp": datetime.now().isoformat(),
        }
        self.save()

    def mark_inaccessible(self, pmid: str, reason: str):
        """Mark a PMID as inaccessible (paywall, error, etc.)."""
        self.data["papers"][str(pmid)] = {
            "status": "inaccessible",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }
        self.save()

    def mark_error(self, pmid: str, error_msg: str):
        """Mark a PMID as failed with an error (will be retried on next run)."""
        self.data["papers"][str(pmid)] = {
            "status": "error",
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        }
        self.save()

    def save_search_results(self, subunit: str, pmids: list[str]):
        """Cache search results for a subunit so we don't re-search."""
        self.data["search_results"][subunit] = {
            "pmids": pmids,
            "timestamp": datetime.now().isoformat(),
        }
        self.save()

    def get_cached_search(self, subunit: str) -> list[str] | None:
        """Get cached search results for a subunit, or None if not cached."""
        entry = self.data["search_results"].get(subunit)
        if entry:
            return entry["pmids"]
        return None

    def get_stats(self) -> dict:
        """Return summary statistics of the checkpoint."""
        papers = self.data["papers"]
        total = len(papers)
        processed = sum(1 for p in papers.values() if p["status"] == "processed")
        inaccessible = sum(1 for p in papers.values() if p["status"] == "inaccessible")
        errors = sum(1 for p in papers.values() if p["status"] == "error")
        total_entries = sum(
            p.get("entries_found", 0)
            for p in papers.values()
            if p["status"] == "processed"
        )
        return {
            "total_papers": total,
            "processed": processed,
            "inaccessible": inaccessible,
            "errors": errors,
            "total_entries_extracted": total_entries,
        }

    def reset(self):
        """Clear all checkpoint data."""
        self.data = {"papers": {}, "search_results": {}}
        self.save()
