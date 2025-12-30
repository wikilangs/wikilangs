#!/usr/bin/env python3
"""
Generate language-to-country mapping for the world map visualization.

This script creates a JSON file mapping ISO 639 language codes to ISO 3166-1
country codes, identifying macrolanguages vs endemic languages.

Requires: pip install pycountry langcodes
"""

import json
from pathlib import Path
from collections import defaultdict

# Known macrolanguages - languages spoken in many countries as a major language
MACROLANGUAGES = {
    'en', 'es', 'fr', 'pt', 'ar', 'zh', 'ru', 'de', 'hi', 'bn',
    'ja', 'ko', 'it', 'nl', 'pl', 'tr', 'vi', 'th', 'id', 'ms',
    'sw', 'fa', 'ur', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa',
    'he', 'el', 'cs', 'hu', 'ro', 'bg', 'uk', 'sr', 'hr', 'sk',
    'sl', 'lt', 'lv', 'et', 'fi', 'sv', 'no', 'da', 'is'
}

# Comprehensive language-to-country mapping
# Format: language_code -> [(country_code, is_primary)]
# is_primary indicates if it's the main/official language
LANGUAGE_COUNTRIES = {
    # Major world languages
    'en': [('US', True), ('GB', True), ('CA', True), ('AU', True), ('NZ', True), ('IE', True), ('ZA', True), ('IN', False), ('PH', False), ('SG', False), ('MY', False), ('NG', False), ('KE', False), ('GH', False), ('JM', False), ('TT', False), ('MT', False)],
    'es': [('ES', True), ('MX', True), ('AR', True), ('CO', True), ('PE', True), ('VE', True), ('CL', True), ('EC', True), ('GT', True), ('CU', True), ('BO', True), ('DO', True), ('HN', True), ('SV', True), ('NI', True), ('CR', True), ('PA', True), ('PY', True), ('UY', True), ('PR', True), ('GQ', True)],
    'fr': [('FR', True), ('CA', False), ('BE', True), ('CH', False), ('SN', True), ('CI', True), ('CM', True), ('CD', True), ('MG', True), ('ML', True), ('BF', True), ('NE', True), ('TG', True), ('BJ', True), ('GA', True), ('CG', True), ('DJ', True), ('RW', True), ('BI', True), ('HT', True), ('LU', True), ('MC', True)],
    'pt': [('PT', True), ('BR', True), ('AO', True), ('MZ', True), ('GW', True), ('TL', True), ('CV', True), ('ST', True)],
    'de': [('DE', True), ('AT', True), ('CH', False), ('LI', True), ('LU', False), ('BE', False)],
    'it': [('IT', True), ('CH', False), ('SM', True), ('VA', True)],
    'nl': [('NL', True), ('BE', False), ('SR', True)],
    'ru': [('RU', True), ('BY', True), ('KZ', False), ('KG', False), ('UA', False)],
    'zh': [('CN', True), ('TW', True), ('HK', True), ('MO', True), ('SG', False), ('MY', False)],
    'ja': [('JP', True)],
    'ko': [('KR', True), ('KP', True)],
    'ar': [('SA', True), ('EG', True), ('IQ', True), ('MA', True), ('DZ', True), ('SD', True), ('SY', True), ('TN', True), ('YE', True), ('JO', True), ('AE', True), ('LB', True), ('LY', True), ('KW', True), ('OM', True), ('QA', True), ('BH', True), ('PS', True), ('MR', True), ('SO', True), ('DJ', False)],
    'hi': [('IN', True)],
    'bn': [('BD', True), ('IN', False)],
    'pa': [('IN', False), ('PK', False)],
    'ta': [('IN', False), ('LK', False), ('SG', False), ('MY', False)],
    'te': [('IN', False)],
    'mr': [('IN', False)],
    'gu': [('IN', False)],
    'kn': [('IN', False)],
    'ml': [('IN', False)],
    'or': [('IN', False)],
    'ur': [('PK', True), ('IN', False)],
    'fa': [('IR', True), ('AF', False), ('TJ', False)],
    'tr': [('TR', True), ('CY', False)],
    'vi': [('VN', True)],
    'th': [('TH', True)],
    'id': [('ID', True)],
    'ms': [('MY', True), ('BN', True), ('SG', False), ('ID', False)],
    'tl': [('PH', True)],
    'sw': [('TZ', True), ('KE', True), ('UG', False), ('RW', False), ('CD', False)],
    'am': [('ET', True)],
    'ha': [('NG', False), ('NE', False), ('GH', False)],
    'yo': [('NG', False), ('BJ', False)],
    'ig': [('NG', False)],
    'zu': [('ZA', False)],
    'xh': [('ZA', False)],
    'af': [('ZA', True), ('NA', False)],
    'st': [('ZA', False), ('LS', True)],
    'tn': [('ZA', False), ('BW', True)],
    'ts': [('ZA', False)],
    'ss': [('SZ', True), ('ZA', False)],
    've': [('ZA', False)],
    'nr': [('ZA', False)],
    'nso': [('ZA', False)],

    # European languages
    'pl': [('PL', True)],
    'uk': [('UA', True)],
    'cs': [('CZ', True)],
    'sk': [('SK', True)],
    'hu': [('HU', True), ('RO', False), ('SK', False), ('RS', False)],
    'ro': [('RO', True), ('MD', True)],
    'bg': [('BG', True)],
    'sr': [('RS', True), ('BA', False), ('ME', False)],
    'hr': [('HR', True), ('BA', False)],
    'sl': [('SI', True)],
    'mk': [('MK', True)],
    'sq': [('AL', True), ('XK', True), ('MK', False)],
    'el': [('GR', True), ('CY', True)],
    'fi': [('FI', True)],
    'sv': [('SE', True), ('FI', False)],
    'no': [('NO', True)],
    'da': [('DK', True)],
    'is': [('IS', True)],
    'et': [('EE', True)],
    'lv': [('LV', True)],
    'lt': [('LT', True)],
    'be': [('BY', True)],
    'mt': [('MT', True)],
    'ga': [('IE', True)],
    'cy': [('GB', False)],
    'gd': [('GB', False)],
    'br': [('FR', False)],
    'oc': [('FR', False), ('ES', False), ('IT', False)],
    'ca': [('ES', False), ('AD', True), ('FR', False), ('IT', False)],
    'gl': [('ES', False)],
    'eu': [('ES', False), ('FR', False)],
    'lb': [('LU', True)],
    'fy': [('NL', False)],

    # Central/South Asian
    'he': [('IL', True)],
    'yi': [('IL', False), ('US', False)],
    'ka': [('GE', True)],
    'hy': [('AM', True)],
    'az': [('AZ', True), ('IR', False)],
    'kk': [('KZ', True)],
    'uz': [('UZ', True)],
    'tg': [('TJ', True)],
    'ky': [('KG', True)],
    'tk': [('TM', True)],
    'mn': [('MN', True)],
    'bo': [('CN', False), ('IN', False)],
    'ne': [('NP', True), ('IN', False)],
    'si': [('LK', True)],
    'dz': [('BT', True)],
    'my': [('MM', True)],
    'km': [('KH', True)],
    'lo': [('LA', True)],

    # Southeast Asian & Pacific
    'jv': [('ID', False)],
    'su': [('ID', False)],
    'ceb': [('PH', False)],
    'war': [('PH', False)],
    'ilo': [('PH', False)],
    'hil': [('PH', False)],
    'pam': [('PH', False)],
    'bcl': [('PH', False)],
    'mi': [('NZ', True)],
    'sm': [('WS', True), ('AS', True)],
    'to': [('TO', True)],
    'fj': [('FJ', True)],
    'haw': [('US', False)],

    # Other
    'eo': [],  # Esperanto - no specific country
    'ia': [],  # Interlingua
    'vo': [],  # Volapük
    'io': [],  # Ido
    'la': [('VA', True)],  # Latin
    'grc': [('GR', False)],  # Ancient Greek
    'sa': [('IN', False)],  # Sanskrit
    'pi': [('IN', False), ('LK', False)],  # Pali

    # More languages with country mappings
    'sco': [('GB', False)],  # Scots
    'nds': [('DE', False), ('NL', False)],  # Low German
    'bar': [('DE', False), ('AT', False)],  # Bavarian
    'als': [('DE', False), ('CH', False), ('FR', False)],  # Alemannic
    'li': [('NL', False), ('BE', False)],  # Limburgish
    'wa': [('BE', False)],  # Walloon
    'co': [('FR', False)],  # Corsican
    'sc': [('IT', False)],  # Sardinian
    'rm': [('CH', False)],  # Romansh
    'fur': [('IT', False)],  # Friulian
    'lmo': [('IT', False), ('CH', False)],  # Lombard
    'pms': [('IT', False)],  # Piedmontese
    'vec': [('IT', False)],  # Venetian
    'scn': [('IT', False)],  # Sicilian
    'nap': [('IT', False)],  # Neapolitan
    'lij': [('IT', False)],  # Ligurian
    'eml': [('IT', False)],  # Emilian-Romagnol
    'ast': [('ES', False)],  # Asturian
    'an': [('ES', False)],  # Aragonese
    'ext': [('ES', False)],  # Extremaduran
    'hsb': [('DE', False)],  # Upper Sorbian
    'dsb': [('DE', False)],  # Lower Sorbian
    'csb': [('PL', False)],  # Kashubian
    'szl': [('PL', False)],  # Silesian
    'rue': [('UA', False), ('SK', False)],  # Rusyn

    # African languages
    'rw': [('RW', True)],  # Kinyarwanda
    'rn': [('BI', True)],  # Kirundi
    'lg': [('UG', False)],  # Luganda
    'sn': [('ZW', True)],  # Shona
    'ny': [('MW', True), ('ZM', False), ('MZ', False)],  # Chichewa
    'wo': [('SN', False), ('GM', False)],  # Wolof
    'ff': [('SN', False), ('NG', False), ('GN', False), ('MR', False)],  # Fulah
    'bm': [('ML', False)],  # Bambara
    'ee': [('GH', False), ('TG', False)],  # Ewe
    'tw': [('GH', False)],  # Twi
    'ak': [('GH', False)],  # Akan
    'ti': [('ER', True), ('ET', False)],  # Tigrinya
    'so': [('SO', True), ('ET', False), ('DJ', False), ('KE', False)],  # Somali
    'om': [('ET', False)],  # Oromo
    'mg': [('MG', True)],  # Malagasy
    'ln': [('CD', False), ('CG', False)],  # Lingala
    'kg': [('CD', False), ('CG', False), ('AO', False)],  # Kongo

    # Native American
    'nv': [('US', False)],  # Navajo
    'chr': [('US', False)],  # Cherokee
    'chy': [('US', False)],  # Cheyenne
    'cr': [('CA', False)],  # Cree
    'oj': [('CA', False), ('US', False)],  # Ojibwe
    'qu': [('PE', True), ('BO', True), ('EC', False)],  # Quechua
    'ay': [('BO', True), ('PE', False)],  # Aymara
    'gn': [('PY', True), ('BO', False), ('AR', False), ('BR', False)],  # Guarani
    'nah': [('MX', False)],  # Nahuatl
}

def generate_country_language_data():
    """Generate the mapping data for the world map."""

    # Load existing languages
    languages_file = Path(__file__).parent.parent / 'website' / 'src' / 'data' / 'languages.json'
    with open(languages_file, 'r', encoding='utf-8') as f:
        languages = json.load(f)

    available_codes = {lang['code'] for lang in languages if lang.get('has_models')}

    # Build country -> languages mapping
    country_languages = defaultdict(lambda: {'endemic': [], 'macro': []})

    for lang_code, countries in LANGUAGE_COUNTRIES.items():
        if lang_code not in available_codes:
            continue

        is_macro = lang_code in MACROLANGUAGES

        for country_code, is_primary in countries:
            if is_macro:
                country_languages[country_code]['macro'].append({
                    'code': lang_code,
                    'primary': is_primary
                })
            else:
                country_languages[country_code]['endemic'].append({
                    'code': lang_code,
                    'primary': is_primary
                })

    # Build language -> countries mapping (for reverse lookup)
    language_countries = {}
    for lang_code, countries in LANGUAGE_COUNTRIES.items():
        if lang_code in available_codes:
            language_countries[lang_code] = {
                'countries': [c[0] for c in countries],
                'is_macro': lang_code in MACROLANGUAGES
            }

    # Calculate stats per country
    country_stats = {}
    for country_code, data in country_languages.items():
        total = len(data['endemic']) + len(data['macro'])
        country_stats[country_code] = {
            'total': total,
            'endemic': len(data['endemic']),
            'macro': len(data['macro']),
            'languages': data
        }

    output = {
        'country_languages': dict(country_languages),
        'language_countries': language_countries,
        'country_stats': country_stats,
        'available_language_count': len(available_codes)
    }

    return output

def main():
    output = generate_country_language_data()

    output_file = Path(__file__).parent.parent / 'website' / 'src' / 'data' / 'language_countries.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Generated {output_file}")
    print(f"Countries with languages: {len(output['country_stats'])}")
    print(f"Languages with country mapping: {len(output['language_countries'])}")

if __name__ == '__main__':
    main()
