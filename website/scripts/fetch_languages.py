#!/usr/bin/env python3
"""
Fetch all language data from wikilangs and HuggingFace at build time.

This script:
1. Gets all available languages from wikilangs
2. Fetches model cards (README.md) from each HuggingFace repo
3. Parses YAML frontmatter for metrics
4. Outputs a single languages.json for Astro to consume
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, asdict

# Add parent to path so we can import wikilangs
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp

try:
    import yaml
except ImportError:
    print("Installing pyyaml...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
    import yaml

from wikilangs import languages_with_metadata


@dataclass
class LanguageData:
    """Complete language data for the website."""
    code: str
    name: str
    common_name: Optional[str] = None
    alpha_2: Optional[str] = None
    alpha_3: Optional[str] = None
    scope: Optional[str] = None
    language_type: Optional[str] = None

    # Metrics from model card
    vocabulary_size: Optional[int] = None
    best_compression_ratio: Optional[float] = None
    best_isotropy: Optional[float] = None

    # Content from README
    has_models: bool = False
    model_card_excerpt: Optional[str] = None

    # HuggingFace URLs
    hf_url: str = ""
    visualizations_base: str = ""


def parse_yaml_frontmatter(content: str) -> dict[str, Any]:
    """Extract YAML frontmatter from markdown content."""
    if not content.startswith('---'):
        return {}

    try:
        # Find the closing ---
        end = content.find('---', 3)
        if end == -1:
            return {}

        yaml_content = content[3:end].strip()
        return yaml.safe_load(yaml_content) or {}
    except Exception as e:
        print(f"  Warning: Failed to parse YAML: {e}")
        return {}


def extract_metrics(frontmatter: dict) -> dict[str, Any]:
    """Extract metrics from frontmatter."""
    metrics = {}

    if 'metrics' in frontmatter:
        for metric in frontmatter['metrics']:
            name = metric.get('name', '')
            value = metric.get('value')
            if name and value is not None:
                metrics[name] = value

    return metrics


def extract_excerpt(content: str, max_length: int = 500) -> str:
    """Extract a clean excerpt from the README content."""
    # Remove frontmatter
    if content.startswith('---'):
        end = content.find('---', 3)
        if end != -1:
            content = content[end + 3:].strip()

    # Find first meaningful paragraph (skip headers and images)
    lines = content.split('\n')
    excerpt_lines = []

    for line in lines:
        line = line.strip()
        # Skip empty lines, headers, images, and links
        if not line or line.startswith('#') or line.startswith('!') or line.startswith('['):
            continue
        # Skip table separators
        if line.startswith('|') or line.startswith('-'):
            continue

        excerpt_lines.append(line)
        if len(' '.join(excerpt_lines)) > max_length:
            break

    excerpt = ' '.join(excerpt_lines)[:max_length]
    if len(excerpt) == max_length:
        excerpt = excerpt.rsplit(' ', 1)[0] + '...'

    return excerpt


async def fetch_model_card(session: aiohttp.ClientSession, lang: str) -> Optional[str]:
    """Fetch README.md from HuggingFace repo."""
    url = f"https://huggingface.co/wikilangs/{lang}/raw/main/README.md"

    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                return await resp.text()
            else:
                return None
    except Exception as e:
        print(f"  Warning: Failed to fetch {lang}: {e}")
        return None


async def process_language(
    session: aiohttp.ClientSession,
    lang_info
) -> LanguageData:
    """Process a single language: fetch model card and extract data."""

    data = LanguageData(
        code=lang_info.code,
        name=lang_info.name,
        common_name=lang_info.common_name,
        alpha_2=lang_info.alpha_2,
        alpha_3=lang_info.alpha_3,
        scope=lang_info.scope,
        language_type=getattr(lang_info, 'type', None),
        hf_url=f"https://huggingface.co/wikilangs/{lang_info.code}",
        visualizations_base=f"https://huggingface.co/wikilangs/{lang_info.code}/resolve/main/visualizations"
    )

    # Fetch model card
    content = await fetch_model_card(session, lang_info.code)

    if content:
        data.has_models = True

        # Parse frontmatter
        frontmatter = parse_yaml_frontmatter(content)
        metrics = extract_metrics(frontmatter)

        # Extract known metrics
        data.vocabulary_size = metrics.get('vocabulary_size')
        data.best_compression_ratio = metrics.get('best_compression_ratio')
        data.best_isotropy = metrics.get('best_isotropy')

        # Extract excerpt
        data.model_card_excerpt = extract_excerpt(content)

    return data


async def fetch_all_languages(concurrency: int = 25) -> list[LanguageData]:
    """Fetch data for all languages with controlled concurrency."""

    print("Fetching language list from wikilangs...")
    try:
        lang_infos = languages_with_metadata('latest')
    except Exception as e:
        raise RuntimeError(
            "Failed to retrieve languages from wikilangs (dataset config discovery failed). "
            f"Original error: {e}"
        )

    print(f"Found {len(lang_infos)} languages")

    if len(lang_infos) == 0:
        raise RuntimeError(
            "No languages were discovered (0). This would generate an empty website."
        )

    results: list[LanguageData] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch_with_semaphore(session, lang_info):
        async with semaphore:
            return await process_language(session, lang_info)

    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_with_semaphore(session, lang) for lang in lang_infos]

        # Process with progress
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if completed % 50 == 0 or completed == len(lang_infos):
                print(f"  Processed {completed}/{len(lang_infos)} languages")

    # Sort by code
    results.sort(key=lambda x: x.code)

    return results


def generate_output(languages: list[LanguageData], output_path: Path):
    """Generate the JSON output file."""

    # Convert to dicts
    data = [asdict(lang) for lang in languages]

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(data)} languages to {output_path}")

    # Print summary stats
    with_models = sum(1 for lang in languages if lang.has_models)
    with_metrics = sum(1 for lang in languages if lang.vocabulary_size)

    print(f"\nSummary:")
    print(f"  Total languages: {len(languages)}")
    print(f"  With model cards: {with_models}")
    print(f"  With full metrics: {with_metrics}")


async def main():
    output_path = Path(__file__).parent.parent / "src" / "data" / "languages.json"

    print("=" * 60)
    print("Wikilangs Build: Fetching Language Data")
    print("=" * 60)

    languages = await fetch_all_languages()
    generate_output(languages, output_path)

    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
