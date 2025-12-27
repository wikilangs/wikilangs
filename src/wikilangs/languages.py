"""Languages module for Wikilangs.

This module provides functionality to discover available language codes
for Wikipedia datasets on specific dates, with optional ISO 639 metadata enrichment.
"""

from dataclasses import dataclass
from typing import Optional
 

try:
    import pycountry
except ImportError:
    pycountry = None


@dataclass
class LanguageInfo:
    """Language information with ISO 639 metadata.
    
    Attributes:
        code (str): The language code (as found in the dataset)
        name (str): The official language name from ISO 639
        common_name (str, optional): Common name if different from official name
        alpha_2 (str, optional): ISO 639-1 two-letter code
        alpha_3 (str, optional): ISO 639-3 three-letter code
        scope (str, optional): Language scope (Individual, Macrolanguage, etc.)
        type (str, optional): Language type (Living, Historical, etc.)
        bibliographic (str, optional): ISO 639-2/B bibliographic code
        terminological (str, optional): ISO 639-2/T terminological code
    """
    code: str
    name: str
    common_name: Optional[str] = None
    alpha_2: Optional[str] = None
    alpha_3: Optional[str] = None
    scope: Optional[str] = None
    type: Optional[str] = None
    bibliographic: Optional[str] = None
    terminological: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation showing code and name."""
        if self.common_name and self.common_name != self.name:
            return f"{self.code}: {self.name} ({self.common_name})"
        return f"{self.code}: {self.name}"


def languages(date: str = "latest") -> list[str]:
    """Get available language codes for a given date.

    This function queries the `omarkamali/wikipedia-monthly` dataset configs
    and extracts languages from config names. Configs are of the form
    ``{date}.{lang}`` or ``latest.{lang}``.
    """
    try:
        from datasets import get_dataset_config_names

        configs = get_dataset_config_names("omarkamali/wikipedia-monthly")

        prefix = f"{date}."
        langs = [c.split(".", 1)[1] for c in configs if c.startswith(prefix)]

        # If no languages for the requested date, fall back to latest.*
        if not langs and date != "latest":
            langs = [c.split(".", 1)[1] for c in configs if c.startswith("latest.")]

        if not langs:
            return []

        return sorted(list(set(langs)))

    except Exception as e:
        raise FileNotFoundError(f"Could not retrieve available languages for date {date}: {e}")


def languages_with_metadata(date: str = "latest") -> list[LanguageInfo]:
    """Get available language codes with ISO 639 metadata enrichment.
    
    Args:
        date (str): Date of the dataset (format: YYYYMMDD, default: "latest")
        
    Returns:
        list[LanguageInfo]: List of LanguageInfo objects with ISO 639 metadata
        
    Examples:
        >>> # Get available languages with metadata
        >>> lang_infos = languages_with_metadata(date='20251201')
        >>> for lang_info in lang_infos:
        ...     print(f"{lang_info.code}: {lang_info.name}")
        ary: Standard Moroccan Tamazight
        en: English
        
        >>> # Access specific metadata
        >>> lang_info = lang_infos[0]
        >>> print(f"Alpha-2: {lang_info.alpha_2}")
        >>> print(f"Scope: {lang_info.scope}")
        
    Note:
        Requires pycountry to be installed for metadata enrichment.
        If pycountry is not available, falls back to basic LanguageInfo with just code and name.
    """
    # Get the basic language codes first
    codes = languages(date)
    
    enriched = []
    
    for code in codes:
        if pycountry is None:
            # Fallback when pycountry is not available
            enriched.append(LanguageInfo(code=code, name=code.upper()))
            continue
        
        try:
            # Try to find the language in pycountry database
            # pycountry.languages.lookup() can match various formats
            lang = pycountry.languages.lookup(code)
            
            enriched.append(LanguageInfo(
                code=code,
                name=lang.name,
                common_name=getattr(lang, 'common_name', None),
                alpha_2=getattr(lang, 'alpha_2', None),
                alpha_3=getattr(lang, 'alpha_3', None),
                scope=getattr(lang, 'scope', None),
                type=getattr(lang, 'type', None),
                bibliographic=getattr(lang, 'bibliographic', None),
                terminological=getattr(lang, 'terminological', None)
            ))
            
        except LookupError:
            # Language code not found in ISO 639 database
            # This can happen for regional variants or non-standard codes
            # Create a basic LanguageInfo with the code as the name
            enriched.append(LanguageInfo(
                code=code,
                name=_format_unknown_language_name(code)
            ))
        except Exception:
            # Any other error, create basic LanguageInfo
            enriched.append(LanguageInfo(
                code=code,
                name=_format_unknown_language_name(code)
            ))
    
    return enriched


def _format_unknown_language_name(code: str) -> str:
    """Format a language name for unknown codes.
    
    Args:
        code (str): Language code
        
    Returns:
        str: Formatted language name
    """
    # Try to make a reasonable display name from the code
    if len(code) == 2:
        return f"{code.upper()} (ISO 639-1)"
    elif len(code) == 3:
        return f"{code.upper()} (ISO 639-3)"
    else:
        return f"{code.upper()} (Unknown)"


def get_language_info(code: str) -> Optional[LanguageInfo]:
    """Get ISO 639 metadata for a single language code.
    
    Args:
        code (str): Language code to look up
        
    Returns:
        LanguageInfo or None: Language information if found, None otherwise
        
    Examples:
        >>> info = get_language_info('en')
        >>> print(info.name)  # 'English'
        >>> print(info.alpha_2)  # 'en'
        >>> print(info.alpha_3)  # 'eng'
    """
    if pycountry is None:
        return LanguageInfo(code=code, name=code.upper())
    
    try:
        lang = pycountry.languages.lookup(code)
        return LanguageInfo(
            code=code,
            name=lang.name,
            common_name=getattr(lang, 'common_name', None),
            alpha_2=getattr(lang, 'alpha_2', None),
            alpha_3=getattr(lang, 'alpha_3', None),
            scope=getattr(lang, 'scope', None),
            type=getattr(lang, 'type', None),
            bibliographic=getattr(lang, 'bibliographic', None),
            terminological=getattr(lang, 'terminological', None)
        )
    except LookupError:
        return LanguageInfo(code=code, name=_format_unknown_language_name(code))
    except Exception:
        return None
