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


# Language name overrides for codes with missing/incorrect names
LANGUAGE_NAME_OVERRIDES = {
    # Wikipedia language codes that don't match ISO standards
    "en": "English",
    "simple": "Simple English",
    "cbk-zam": "Chavacano",
    "map-bms": "Banyumasan",
    "nds-nl": "Low Saxon",
    "be-tarask": "Belarusian (Taraškievica)",
    "roa-tara": "Tarantino",
    "zh-yue": "Cantonese",
    "zh-classical": "Classical Chinese",
    "zh-min-nan": "Min Nan",

    # Short codes that need full names
    "ak": "Akan",
    "as": "Assamese",
    "bh": "Bihari",
    "en": "English",
    "ho": "Hiri Motu",
    "ii": "Sichuan Yi",
    "na": "Nauruan",
    "ng": "Ndonga",
    "nr": "Southern Ndebele",
    "nah": "Nahuatl",
    "pi": "Pali",
    "za": "Zhuang",

    # Languages with incorrect metadata names
    "ang": "Old English",
    "arc": "Aramaic",
    "diq": "Zazaki",
    "got": "Gothic",
    "rup": "Aromanian",
    "cu": "Church Slavonic",
    "grc": "Ancient Greek",

    # Common name preferences
    "el": "Greek",
    "hy": "Armenian",
    "ka": "Georgian",
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
    "fa": "Persian",
    "he": "Hebrew",
    "yi": "Yiddish",
    "sq": "Albanian",
    "eu": "Basque",
    "cy": "Welsh",
    "ga": "Irish",
    "gd": "Scottish Gaelic",
    "br": "Breton",
    "kw": "Cornish",
    "gv": "Manx",
    "lb": "Luxembourgish",

    # Remove verbose annotations
    "ia": "Interlingua",
    "oc": "Occitan",
    "war": "Waray",

    # Fix capitalized/short codes
    "eml": "Emilian-Romagnol",
    "fon": "Fon",
    "ha": "Hausa",
    "hu": "Hungarian",
    "ik": "Inupiaq",
    "sa": "Sanskrit",
    "to": "Tongan",
    "wa": "Walloon",
}


def clean_language_name(name: str, code: str) -> str:
    """Clean up language name by removing annotations and fixing formatting."""
    # Use override if available
    if code in LANGUAGE_NAME_OVERRIDES:
        return LANGUAGE_NAME_OVERRIDES[code]

    # Remove common annotations in parentheses
    patterns_to_remove = [
        r'\s*\(ISO 639-\d+\)',           # (ISO 639-1), (ISO 639-3)
        r'\s*\(Unknown\)',                # (Unknown)
        r'\s*\(individual language\)',    # (individual language)
        r'\s*\(macrolanguage\)',          # (macrolanguage)
        r'\s*\(ca\.\s*[\d\-]+\)',         # (ca. 450-1100)
        r'\s*\(\d+[-–]\d+\s*BCE?\)',      # (700-300 BCE)
        r'\s*\(\d+[-–]\d+\)',             # (1453-)
        r'\s*\(post \d+\)',               # (post 1500)
        r'\s*\([^)]*Association[^)]*\)',  # (International Auxiliary Language Association)
        r'\s*\([^)]*Philippines[^)]*\)',  # (Philippines)
    ]

    cleaned = name
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Clean up any double spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # If the name is all uppercase and longer than 2 chars, title case it
    if cleaned.isupper() and len(cleaned) > 2:
        cleaned = cleaned.title()

    # If the name is just the code in uppercase, it's invalid
    if cleaned.upper() == code.upper() or len(cleaned) <= 2:
        # Try to find a better name
        return code.upper()  # Will be fixed by overrides

    return cleaned


@dataclass
class LanguageData:
    """Complete language data for the website."""
    code: str
    name: str
    common_name: Optional[str] = None
    native_name: Optional[str] = None  # Language name in native script
    text_direction: str = "ltr"  # ltr or rtl
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

    # Wikipedia samples (pre-fetched)
    wikipedia_samples: list[str] = None  # type: ignore

    # HuggingFace URLs
    hf_url: str = ""
    visualizations_base: str = ""

    def __post_init__(self):
        if self.wikipedia_samples is None:
            self.wikipedia_samples = []


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


# Cache for native language names (fetched once)
_native_names_cache: dict[str, str] = {}


async def fetch_all_native_names(session: aiohttp.ClientSession) -> dict[str, str]:
    """Fetch all language native names from Wikipedia (cached)."""
    global _native_names_cache
    if _native_names_cache:
        return _native_names_cache

    url = "https://en.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=languages&format=json"
    headers = {"User-Agent": "WikiLangs/1.0 (https://wikilangs.org; wikilangs@omarkama.li)"}
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status == 200:
                data = await resp.json()
                languages = data.get('query', {}).get('languages', [])
                for l in languages:
                    code = l.get('code')
                    name = l.get('*')
                    if code and name:
                        _native_names_cache[code] = name
    except Exception as e:
        print(f"  Warning: Failed to fetch native names: {e}")

    return _native_names_cache


async def fetch_text_direction(session: aiohttp.ClientSession, lang: str) -> str:
    """Fetch text direction from the language's Wikipedia."""
    url = f"https://{lang}.wikipedia.org/w/api.php?action=query&meta=siteinfo&siprop=general&format=json"
    headers = {"User-Agent": "WikiLangs/1.0 (https://wikilangs.org; wikilangs@omarkama.li)"}

    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.json()
                general = data.get('query', {}).get('general', {})
                # 'rtl' key EXISTS (even if empty string) for RTL languages
                if 'rtl' in general:
                    return 'rtl'
    except Exception:
        pass

    return 'ltr'


async def fetch_wikipedia_samples(session: aiohttp.ClientSession, lang: str, num_samples: int = 3) -> list[str]:
    """Fetch random article excerpts from this language's Wikipedia."""
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/random/summary"
    headers = {"User-Agent": "WikiLangs/1.0 (https://wikilangs.org; wikilangs@omarkama.li)"}
    samples = []

    for _ in range(num_samples):
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10), allow_redirects=True) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    extract = data.get('extract', '')
                    if extract:
                        # Extract first sentence (handle multiple sentence-ending punctuation)
                        import re
                        sentences = re.split(r'(?<=[.!?。！？।؟])\s+', extract)
                        for sentence in sentences:
                            if len(sentence.strip()) > 20:
                                samples.append(sentence.strip())
                                break
                elif resp.status == 429:
                    # Rate limited, wait a bit
                    await asyncio.sleep(1)
        except Exception:
            pass

        # Small delay between requests to avoid rate limiting
        await asyncio.sleep(0.2)

    return samples


async def process_language(
    session: aiohttp.ClientSession,
    lang_info,
    native_names: dict[str, str]
) -> LanguageData:
    """Process a single language: fetch model card, text direction, and Wikipedia samples."""

    # Fetch text direction, model card, and Wikipedia samples in parallel
    dir_task = fetch_text_direction(session, lang_info.code)
    model_task = fetch_model_card(session, lang_info.code)
    samples_task = fetch_wikipedia_samples(session, lang_info.code)

    text_direction = await dir_task
    content = await model_task
    wikipedia_samples = await samples_task

    # Get native name from pre-fetched cache
    native_name = native_names.get(lang_info.code)

    # Clean up language names
    cleaned_name = clean_language_name(lang_info.name, lang_info.code)
    cleaned_common_name = None
    if lang_info.common_name:
        cleaned_common_name = clean_language_name(lang_info.common_name, lang_info.code)

    data = LanguageData(
        code=lang_info.code,
        name=cleaned_name,
        common_name=cleaned_common_name,
        native_name=native_name,
        text_direction=text_direction,
        alpha_2=lang_info.alpha_2,
        alpha_3=lang_info.alpha_3,
        scope=lang_info.scope,
        language_type=getattr(lang_info, 'type', None),
        wikipedia_samples=wikipedia_samples,
        hf_url=f"https://huggingface.co/wikilangs/{lang_info.code}",
        visualizations_base=f"https://huggingface.co/wikilangs/{lang_info.code}/resolve/main/visualizations"
    )

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

    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Pre-fetch all native language names from Wikipedia (single request)
        print("  Fetching native language names from Wikipedia...")
        native_names = await fetch_all_native_names(session)
        print(f"  Found {len(native_names)} native names")

        async def fetch_with_semaphore(lang_info):
            async with semaphore:
                return await process_language(session, lang_info, native_names)

        tasks = [fetch_with_semaphore(lang) for lang in lang_infos]

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
    with_samples = sum(1 for lang in languages if lang.wikipedia_samples)
    with_native = sum(1 for lang in languages if lang.native_name)

    print(f"\nSummary:")
    print(f"  Total languages: {len(languages)}")
    print(f"  With model cards: {with_models}")
    print(f"  With full metrics: {with_metrics}")
    print(f"  With Wikipedia samples: {with_samples}")
    print(f"  With native names: {with_native}")


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
