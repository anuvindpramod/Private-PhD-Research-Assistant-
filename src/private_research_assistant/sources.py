from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    name: str
    query: str
    include_domains: tuple[str, ...]
    limit: int


CURATED_SOURCES: tuple[SourceConfig, ...] = (
    SourceConfig(
        source_id="jobs_ac_uk",
        name="jobs.ac.uk",
        query="AI machine learning PhD studentship Europe UK",
        include_domains=("jobs.ac.uk", "www.jobs.ac.uk"),
        limit=8,
    ),
    SourceConfig(
        source_id="findaphd",
        name="FindAPhD",
        query="artificial intelligence machine learning funded PhD Europe UK",
        include_domains=("findaphd.com", "www.findaphd.com"),
        limit=8,
    ),
    SourceConfig(
        source_id="euraxess",
        name="EURAXESS",
        query="AI machine learning PhD positions Europe",
        include_domains=("euraxess.ec.europa.eu",),
        limit=8,
    ),
    SourceConfig(
        source_id="academic_positions",
        name="Academic Positions",
        query="PhD artificial intelligence machine learning Europe",
        include_domains=("academicpositions.com", "www.academicpositions.com"),
        limit=8,
    ),
    SourceConfig(
        source_id="ellis_jobs",
        name="ELLIS jobs",
        query="PhD machine learning artificial intelligence ELLIS Europe",
        include_domains=("ellis.eu", "www.ellis.eu"),
        limit=5,
    ),
    SourceConfig(
        source_id="phdscanner",
        name="PhD Scanner",
        query="artificial intelligence machine learning PhD Europe",
        include_domains=("phdscanner.com", "www.phdscanner.com"),
        limit=8,
    ),
)


def get_sources(selected_ids: list[str] | None = None) -> list[SourceConfig]:
    if not selected_ids:
        return list(CURATED_SOURCES)
    wanted = set(selected_ids)
    sources = [source for source in CURATED_SOURCES if source.source_id in wanted]
    missing = wanted - {source.source_id for source in sources}
    if missing:
        available = ", ".join(source.source_id for source in CURATED_SOURCES)
        raise ValueError(f"Unknown source id(s): {', '.join(sorted(missing))}. Available: {available}")
    return sources

