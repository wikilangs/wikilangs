#!/usr/bin/env python3
"""
Generate language-to-continent mappings using Wikidata and other sources.

This script maps languages to countries and countries to continents.
"""

import asyncio
import json
from pathlib import Path
import sys

try:
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp

# Continent mapping for countries (ISO alpha-2)
COUNTRY_TO_CONTINENT = {
    # Africa
    "DZ": "Africa", "AO": "Africa", "BJ": "Africa", "BW": "Africa", "BF": "Africa",
    "BI": "Africa", "CV": "Africa", "CM": "Africa", "CF": "Africa", "TD": "Africa",
    "KM": "Africa", "CG": "Africa", "CD": "Africa", "CI": "Africa", "DJ": "Africa",
    "EG": "Africa", "GQ": "Africa", "ER": "Africa", "SZ": "Africa", "ET": "Africa",
    "GA": "Africa", "GM": "Africa", "GH": "Africa", "GN": "Africa", "GW": "Africa",
    "KE": "Africa", "LS": "Africa", "LR": "Africa", "LY": "Africa", "MG": "Africa",
    "MW": "Africa", "ML": "Africa", "MR": "Africa", "MU": "Africa", "MA": "Africa",
    "MZ": "Africa", "NA": "Africa", "NE": "Africa", "NG": "Africa", "RW": "Africa",
    "ST": "Africa", "SN": "Africa", "SC": "Africa", "SL": "Africa", "SO": "Africa",
    "ZA": "Africa", "SS": "Africa", "SD": "Africa", "TZ": "Africa", "TG": "Africa",
    "TN": "Africa", "UG": "Africa", "ZM": "Africa", "ZW": "Africa", "RE": "Africa",
    "YT": "Africa", "SH": "Africa", "EH": "Africa",

    # Asia
    "AF": "Asia", "AM": "Asia", "AZ": "Asia", "BH": "Asia", "BD": "Asia",
    "BT": "Asia", "BN": "Asia", "KH": "Asia", "CN": "Asia", "CY": "Asia",
    "GE": "Asia", "HK": "Asia", "IN": "Asia", "ID": "Asia", "IR": "Asia",
    "IQ": "Asia", "IL": "Asia", "JP": "Asia", "JO": "Asia", "KZ": "Asia",
    "KW": "Asia", "KG": "Asia", "LA": "Asia", "LB": "Asia", "MO": "Asia",
    "MY": "Asia", "MV": "Asia", "MN": "Asia", "MM": "Asia", "NP": "Asia",
    "KP": "Asia", "OM": "Asia", "PK": "Asia", "PS": "Asia", "PH": "Asia",
    "QA": "Asia", "SA": "Asia", "SG": "Asia", "KR": "Asia", "LK": "Asia",
    "SY": "Asia", "TW": "Asia", "TJ": "Asia", "TH": "Asia", "TL": "Asia",
    "TR": "Asia", "TM": "Asia", "AE": "Asia", "UZ": "Asia", "VN": "Asia",
    "YE": "Asia",

    # Europe
    "AL": "Europe", "AD": "Europe", "AT": "Europe", "BY": "Europe", "BE": "Europe",
    "BA": "Europe", "BG": "Europe", "HR": "Europe", "CZ": "Europe", "DK": "Europe",
    "EE": "Europe", "FO": "Europe", "FI": "Europe", "FR": "Europe", "DE": "Europe",
    "GI": "Europe", "GR": "Europe", "GL": "Europe", "GG": "Europe", "HU": "Europe",
    "IS": "Europe", "IE": "Europe", "IM": "Europe", "IT": "Europe", "JE": "Europe",
    "XK": "Europe", "LV": "Europe", "LI": "Europe", "LT": "Europe", "LU": "Europe",
    "MK": "Europe", "MT": "Europe", "MD": "Europe", "MC": "Europe", "ME": "Europe",
    "NL": "Europe", "NO": "Europe", "PL": "Europe", "PT": "Europe", "RO": "Europe",
    "RU": "Europe", "SM": "Europe", "RS": "Europe", "SK": "Europe", "SI": "Europe",
    "ES": "Europe", "SE": "Europe", "CH": "Europe", "UA": "Europe", "GB": "Europe",
    "VA": "Europe", "AX": "Europe",

    # North America
    "AG": "North America", "BS": "North America", "BB": "North America", "BZ": "North America",
    "CA": "North America", "CR": "North America", "CU": "North America", "DM": "North America",
    "DO": "North America", "SV": "North America", "GD": "North America", "GT": "North America",
    "HT": "North America", "HN": "North America", "JM": "North America", "MX": "North America",
    "NI": "North America", "PA": "North America", "PR": "North America", "KN": "North America",
    "LC": "North America", "VC": "North America", "TT": "North America", "US": "North America",
    "VI": "North America", "GP": "North America", "MQ": "North America", "BM": "North America",
    "KY": "North America", "AW": "North America", "CW": "North America", "SX": "North America",

    # South America
    "AR": "South America", "BO": "South America", "BR": "South America", "CL": "South America",
    "CO": "South America", "EC": "South America", "FK": "South America", "GF": "South America",
    "GY": "South America", "PY": "South America", "PE": "South America", "SR": "South America",
    "UY": "South America", "VE": "South America",

    # Oceania
    "AS": "Oceania", "AU": "Oceania", "CK": "Oceania", "FJ": "Oceania", "PF": "Oceania",
    "GU": "Oceania", "KI": "Oceania", "MH": "Oceania", "FM": "Oceania", "NR": "Oceania",
    "NC": "Oceania", "NZ": "Oceania", "NU": "Oceania", "NF": "Oceania", "MP": "Oceania",
    "PW": "Oceania", "PG": "Oceania", "PN": "Oceania", "WS": "Oceania", "SB": "Oceania",
    "TK": "Oceania", "TO": "Oceania", "TV": "Oceania", "VU": "Oceania", "WF": "Oceania",
}

# Direct language-to-continent mappings for languages where the region is well-known
# This is a fallback when Wikidata queries don't return useful info
KNOWN_LANGUAGE_CONTINENTS = {
    # Major world languages (primary region)
    "en": ["Europe", "North America", "Oceania"],
    "es": ["Europe", "South America", "North America"],
    "fr": ["Europe", "Africa"],
    "de": ["Europe"],
    "it": ["Europe"],
    "pt": ["Europe", "South America", "Africa"],
    "ru": ["Europe", "Asia"],
    "zh": ["Asia"],
    "ja": ["Asia"],
    "ko": ["Asia"],
    "ar": ["Africa", "Asia"],
    "hi": ["Asia"],
    "bn": ["Asia"],
    "pa": ["Asia"],
    "ta": ["Asia"],
    "te": ["Asia"],
    "mr": ["Asia"],
    "gu": ["Asia"],
    "ur": ["Asia"],
    "fa": ["Asia"],
    "tr": ["Europe", "Asia"],
    "vi": ["Asia"],
    "th": ["Asia"],
    "id": ["Asia"],
    "ms": ["Asia"],
    "tl": ["Asia"],
    "sw": ["Africa"],
    "ha": ["Africa"],
    "yo": ["Africa"],
    "ig": ["Africa"],
    "am": ["Africa"],
    "so": ["Africa"],
    "zu": ["Africa"],
    "xh": ["Africa"],
    "af": ["Africa"],
    "nl": ["Europe"],
    "pl": ["Europe"],
    "uk": ["Europe"],
    "cs": ["Europe"],
    "el": ["Europe"],
    "hu": ["Europe"],
    "sv": ["Europe"],
    "da": ["Europe"],
    "fi": ["Europe"],
    "no": ["Europe"],
    "he": ["Asia"],
    "yi": ["Europe"],

    # Regional languages by continent
    # Africa
    "aa": ["Africa"],  # Afar
    "ak": ["Africa"],  # Akan
    "bm": ["Africa"],  # Bambara
    "ee": ["Africa"],  # Ewe
    "ff": ["Africa"],  # Fulah
    "lg": ["Africa"],  # Ganda
    "ki": ["Africa"],  # Kikuyu
    "rw": ["Africa"],  # Kinyarwanda
    "ln": ["Africa"],  # Lingala
    "ny": ["Africa"],  # Nyanja
    "om": ["Africa"],  # Oromo
    "rn": ["Africa"],  # Rundi
    "sn": ["Africa"],  # Shona
    "st": ["Africa"],  # Southern Sotho
    "tn": ["Africa"],  # Tswana
    "ts": ["Africa"],  # Tsonga
    "tw": ["Africa"],  # Twi
    "ve": ["Africa"],  # Venda
    "wo": ["Africa"],  # Wolof
    "nso": ["Africa"], # Northern Sotho
    "ss": ["Africa"],  # Swati
    "ti": ["Africa"],  # Tigrinya
    "ary": ["Africa"], # Moroccan Arabic
    "arz": ["Africa"], # Egyptian Arabic
    "kab": ["Africa"], # Kabyle
    "zgh": ["Africa"], # Standard Moroccan Tamazight
    "ber": ["Africa"], # Berber
    "shi": ["Africa"], # Tashelhit
    "tzm": ["Africa"], # Central Atlas Tamazight
    "din": ["Africa"], # Dinka
    "luo": ["Africa"], # Luo
    "gur": ["Africa"], # Frafra
    "dag": ["Africa"], # Dagbani
    "ada": ["Africa"], # Adangme
    "fat": ["Africa"], # Fanti
    "nyn": ["Africa"], # Nyankole
    "toi": ["Africa"], # Tonga
    "bem": ["Africa"], # Bemba
    "loz": ["Africa"], # Lozi
    "umb": ["Africa"], # Umbundu
    "kmb": ["Africa"], # Kimbundu
    "kon": ["Africa"], # Kongo
    "lua": ["Africa"], # Luba-Lulua
    "lub": ["Africa"], # Luba-Katanga

    # Asia
    "ab": ["Asia"],    # Abkhazian
    "ady": ["Asia"],   # Adyghe
    "av": ["Asia"],    # Avaric
    "az": ["Asia"],    # Azerbaijani
    "azb": ["Asia"],   # South Azerbaijani
    "ba": ["Asia"],    # Bashkir
    "ce": ["Asia"],    # Chechen
    "cv": ["Asia"],    # Chuvash
    "ka": ["Asia"],    # Georgian
    "hy": ["Asia"],    # Armenian
    "kk": ["Asia"],    # Kazakh
    "ky": ["Asia"],    # Kyrgyz
    "mn": ["Asia"],    # Mongolian
    "tg": ["Asia"],    # Tajik
    "tk": ["Asia"],    # Turkmen
    "tt": ["Asia"],    # Tatar
    "ug": ["Asia"],    # Uyghur
    "uz": ["Asia"],    # Uzbek
    "bo": ["Asia"],    # Tibetan
    "dz": ["Asia"],    # Dzongkha
    "ne": ["Asia"],    # Nepali
    "si": ["Asia"],    # Sinhala
    "my": ["Asia"],    # Burmese
    "km": ["Asia"],    # Khmer
    "lo": ["Asia"],    # Lao
    "jv": ["Asia"],    # Javanese
    "su": ["Asia"],    # Sundanese
    "min": ["Asia"],   # Minangkabau
    "ace": ["Asia"],   # Acehnese
    "ban": ["Asia"],   # Balinese
    "bug": ["Asia"],   # Buginese
    "bbc": ["Asia"],   # Batak Toba
    "mad": ["Asia"],   # Madurese
    "ilo": ["Asia"],   # Ilocano
    "ceb": ["Asia"],   # Cebuano
    "war": ["Asia"],   # Waray
    "pam": ["Asia"],   # Pampanga
    "bcl": ["Asia"],   # Central Bikol
    "hil": ["Asia"],   # Hiligaynon
    "pag": ["Asia"],   # Pangasinan
    "yue": ["Asia"],   # Cantonese
    "wuu": ["Asia"],   # Wu Chinese
    "nan": ["Asia"],   # Min Nan Chinese
    "hak": ["Asia"],   # Hakka Chinese
    "gan": ["Asia"],   # Gan Chinese
    "hsn": ["Asia"],   # Xiang Chinese
    "cdo": ["Asia"],   # Min Dong Chinese
    "mai": ["Asia"],   # Maithili
    "bho": ["Asia"],   # Bhojpuri
    "awa": ["Asia"],   # Awadhi
    "mag": ["Asia"],   # Magahi
    "new": ["Asia"],   # Newari
    "sa": ["Asia"],    # Sanskrit
    "or": ["Asia"],    # Odia
    "as": ["Asia"],    # Assamese
    "ks": ["Asia"],    # Kashmiri
    "sd": ["Asia"],    # Sindhi
    "doi": ["Asia"],   # Dogri
    "mni": ["Asia"],   # Manipuri
    "sat": ["Asia"],   # Santali
    "kn": ["Asia"],    # Kannada
    "ml": ["Asia"],    # Malayalam
    "dv": ["Asia"],    # Dhivehi
    "ps": ["Asia"],    # Pashto
    "ku": ["Asia"],    # Kurdish
    "ckb": ["Asia"],   # Central Kurdish
    "arc": ["Asia"],   # Aramaic
    "syr": ["Asia"],   # Syriac
    "pnb": ["Asia"],   # Western Punjabi
    "skr": ["Asia"],   # Saraiki
    "lah": ["Asia"],   # Lahnda
    "glk": ["Asia"],   # Gilaki
    "mzn": ["Asia"],   # Mazanderani
    "tly": ["Asia"],   # Talysh
    "lez": ["Asia"],   # Lezghian
    "kbd": ["Asia"],   # Kabardian
    "krc": ["Asia"],   # Karachay-Balkar
    "inh": ["Asia"],   # Ingush
    "os": ["Asia"],    # Ossetic
    "xmf": ["Asia"],   # Mingrelian
    "lzz": ["Asia"],   # Laz
    "alt": ["Asia"],   # Southern Altai
    "tyv": ["Asia"],   # Tuvinian
    "kjh": ["Asia"],   # Khakas
    "xal": ["Asia"],   # Kalmyk
    "bua": ["Asia"],   # Buriat
    "sah": ["Asia"],   # Yakut
    "evn": ["Asia"],   # Evenki
    "gld": ["Asia"],   # Nanai
    "udm": ["Asia"],   # Udmurt
    "koi": ["Asia"],   # Komi-Permyak
    "kv": ["Asia"],    # Komi
    "mhr": ["Asia"],   # Eastern Mari
    "mrj": ["Asia"],   # Western Mari
    "myv": ["Asia"],   # Erzya
    "mdf": ["Asia"],   # Moksha
    "ami": ["Asia"],   # Amis
    "tay": ["Asia"],   # Atayal

    # Europe
    "ang": ["Europe"], # Old English
    "ast": ["Europe"], # Asturian
    "bar": ["Europe"], # Bavarian
    "br": ["Europe"],  # Breton
    "ca": ["Europe"],  # Catalan
    "co": ["Europe"],  # Corsican
    "cy": ["Europe"],  # Welsh
    "eu": ["Europe"],  # Basque
    "ga": ["Europe"],  # Irish
    "gd": ["Europe"],  # Scottish Gaelic
    "gl": ["Europe"],  # Galician
    "gv": ["Europe"],  # Manx
    "hsb": ["Europe"], # Upper Sorbian
    "dsb": ["Europe"], # Lower Sorbian
    "is": ["Europe"],  # Icelandic
    "kw": ["Europe"],  # Cornish
    "la": ["Europe"],  # Latin
    "lb": ["Europe"],  # Luxembourgish
    "li": ["Europe"],  # Limburgish
    "lt": ["Europe"],  # Lithuanian
    "lv": ["Europe"],  # Latvian
    "mk": ["Europe"],  # Macedonian
    "mt": ["Europe"],  # Maltese
    "oc": ["Europe"],  # Occitan
    "rm": ["Europe"],  # Romansh
    "ro": ["Europe"],  # Romanian
    "sc": ["Europe"],  # Sardinian
    "sk": ["Europe"],  # Slovak
    "sl": ["Europe"],  # Slovenian
    "sq": ["Europe"],  # Albanian
    "sr": ["Europe"],  # Serbian
    "bs": ["Europe"],  # Bosnian
    "hr": ["Europe"],  # Croatian
    "et": ["Europe"],  # Estonian
    "fo": ["Europe"],  # Faroese
    "fy": ["Europe"],  # Western Frisian
    "frr": ["Europe"], # Northern Frisian
    "stq": ["Europe"], # Saterland Frisian
    "nds": ["Europe"], # Low German
    "pdc": ["Europe"], # Pennsylvania German
    "ksh": ["Europe"], # Colognian
    "pfl": ["Europe"], # Palatinate German
    "als": ["Europe"], # Alemannic
    "gsw": ["Europe"], # Swiss German
    "nrm": ["Europe"], # Norman
    "pcd": ["Europe"], # Picard
    "wa": ["Europe"],  # Walloon
    "vls": ["Europe"], # West Flemish
    "zea": ["Europe"], # Zeelandic
    "vec": ["Europe"], # Venetian
    "scn": ["Europe"], # Sicilian
    "nap": ["Europe"], # Neapolitan
    "lmo": ["Europe"], # Lombard
    "pms": ["Europe"], # Piedmontese
    "lij": ["Europe"], # Ligurian
    "eml": ["Europe"], # Emilian-Romagnol
    "fur": ["Europe"], # Friulian
    "lad": ["Europe"], # Ladino
    "an": ["Europe"],  # Aragonese
    "ext": ["Europe"], # Extremaduran
    "mwl": ["Europe"], # Mirandese
    "csb": ["Europe"], # Kashubian
    "szl": ["Europe"], # Silesian
    "rue": ["Europe"], # Rusyn
    "be": ["Europe"],  # Belarusian
    "bg": ["Europe"],  # Bulgarian
    "cu": ["Europe"],  # Church Slavic
    "sgs": ["Europe"], # Samogitian
    "ltg": ["Europe"], # Latgalian
    "vro": ["Europe"], # Voro
    "liv": ["Europe"], # Livonian
    "vep": ["Europe"], # Veps
    "izh": ["Europe"], # Ingrian
    "olo": ["Europe"], # Livvi-Karelian
    "krl": ["Europe"], # Karelian
    "se": ["Europe"],  # Northern Sami
    "smn": ["Europe"], # Inari Sami
    "sms": ["Europe"], # Skolt Sami
    "sma": ["Europe"], # Southern Sami
    "smj": ["Europe"], # Lule Sami
    "nn": ["Europe"],  # Norwegian Nynorsk
    "nb": ["Europe"],  # Norwegian Bokmal
    "jbo": ["Europe"], # Lojban (constructed)
    "eo": ["Europe"],  # Esperanto
    "ia": ["Europe"],  # Interlingua
    "ie": ["Europe"],  # Interlingue
    "io": ["Europe"],  # Ido
    "vo": ["Europe"],  # Volapuk
    "nov": ["Europe"], # Novial
    "avk": ["Europe"], # Kotava (constructed, placed in Europe)

    # North America
    "chr": ["North America"], # Cherokee
    "nv": ["North America"],  # Navajo
    "cr": ["North America"],  # Cree
    "oj": ["North America"],  # Ojibwe
    "iu": ["North America"],  # Inuktitut
    "ik": ["North America"],  # Inupiaq
    "kl": ["North America"],  # Kalaallisut
    "atj": ["North America"], # Atikamekw
    "crk": ["North America"], # Plains Cree
    "nah": ["North America"], # Nahuatl
    "yua": ["North America"], # Yucatec Maya
    "myn": ["North America"], # Maya
    "chy": ["North America"], # Cheyenne
    "mus": ["North America"], # Creek
    "srn": ["North America"], # Sranan Tongo
    "gcr": ["North America"], # Guianese Creole
    "ht": ["North America"],  # Haitian Creole
    "jam": ["North America"], # Jamaican Creole
    "pap": ["North America"], # Papiamento

    # South America
    "gn": ["South America"],  # Guarani
    "qu": ["South America"],  # Quechua
    "ay": ["South America"],  # Aymara
    "srm": ["South America"], # Sranan Tongo
    "pbb": ["South America"], # Páez

    # Oceania
    "mi": ["Oceania"],  # Maori
    "sm": ["Oceania"],  # Samoan
    "to": ["Oceania"],  # Tongan
    "fj": ["Oceania"],  # Fijian
    "ty": ["Oceania"],  # Tahitian
    "haw": ["Oceania"], # Hawaiian
    "mh": ["Oceania"],  # Marshallese
    "na": ["Oceania"],  # Nauru
    "ch": ["Oceania"],  # Chamorro
    "pau": ["Oceania"], # Palauan
    "tpi": ["Oceania"], # Tok Pisin
    "bi": ["Oceania"],  # Bislama
    "pih": ["Oceania"], # Pitcairn-Norfolk
    "tet": ["Oceania"], # Tetum
    "niu": ["Oceania"], # Niuean
    "tvl": ["Oceania"], # Tuvalu
    "gil": ["Oceania"], # Gilbertese

    # Additional mappings for remaining languages
    # Africa
    "ann": ["Africa"],    # Obolo (Nigeria)
    "fon": ["Africa"],    # Fon (Benin)
    "gpe": ["Africa"],    # Ghanaian Pidgin English
    "guw": ["Africa"],    # Gun (Benin)
    "hz": ["Africa"],     # Herero (Namibia)
    "igl": ["Africa"],    # Igala (Nigeria)
    "kbp": ["Africa"],    # Kabiye (Togo)
    "kcg": ["Africa"],    # Tyap (Nigeria)
    "kg": ["Africa"],     # Kongo (Congo)
    "kj": ["Africa"],     # Kuanyama (Namibia/Angola)
    "knc": ["Africa"],    # Central Kanuri (Nigeria)
    "kr": ["Africa"],     # Kanuri (Nigeria)
    "kus": ["Africa"],    # Kusaal (Ghana)
    "mg": ["Africa"],     # Malagasy (Madagascar)
    "mos": ["Africa"],    # Mossi (Burkina Faso)
    "ng": ["Africa"],     # Ndonga (Namibia)
    "nqo": ["Africa"],    # N'Ko (West Africa)
    "nr": ["Africa"],     # South Ndebele (South Africa)
    "nup": ["Africa"],    # Nupe (Nigeria)
    "pcm": ["Africa"],    # Nigerian Pidgin
    "sg": ["Africa"],     # Sango (Central African Republic)
    "tig": ["Africa"],    # Tigre (Eritrea)
    "tum": ["Africa"],    # Tumbuka (Malawi/Zambia)
    "dga": ["Africa"],    # Southern Dagaare (Ghana)

    # Asia
    "anp": ["Asia"],      # Angika (India/Nepal)
    "bdr": ["Asia"],      # West Coast Bajau (Malaysia)
    "bew": ["Asia"],      # Betawi (Indonesia)
    "bh": ["Asia"],       # Bihari (India) - ISO 639-1
    "bjn": ["Asia"],      # Banjar (Indonesia)
    "blk": ["Asia"],      # Pa'o Karen (Myanmar)
    "bpy": ["Asia"],      # Bishnupriya (India/Bangladesh)
    "btm": ["Asia"],      # Batak Mandailing (Indonesia)
    "bxr": ["Asia"],      # Russia Buriat (Russia/Mongolia)
    "crh": ["Asia", "Europe"],  # Crimean Tatar
    "diq": ["Asia"],      # Dimli/Zazaki (Turkey)
    "dtp": ["Asia"],      # Kadazan Dusun (Malaysia)
    "dty": ["Asia"],      # Dotyali (Nepal)
    "gom": ["Asia"],      # Goan Konkani (India)
    "gor": ["Asia"],      # Gorontalo (Indonesia)
    "guc": ["South America"],  # Wayuu (Colombia/Venezuela)
    "hif": ["Oceania"],   # Fiji Hindi (Fiji)
    "ho": ["Asia"],       # Ho (India)
    "hyw": ["Asia"],      # Western Armenian
    "iba": ["Asia"],      # Iban (Malaysia)
    "ii": ["Asia"],       # Sichuan Yi (China)
    "kaa": ["Asia"],      # Kara-Kalpak (Uzbekistan)
    "kge": ["Asia"],      # Komering (Indonesia)
    "lbe": ["Asia"],      # Lak (Russia - Dagestan)
    "lrc": ["Asia"],      # Northern Luri (Iran)
    "lzh": ["Asia"],      # Literary Chinese
    "mnw": ["Asia"],      # Mon (Myanmar/Thailand)
    "nia": ["Asia"],      # Nias (Indonesia)
    "pi": ["Asia"],       # Pali (ancient Buddhist language)
    "pnt": ["Asia", "Europe"],  # Pontic Greek
    "pwn": ["Asia"],      # Paiwan (Taiwan)
    "rki": ["Asia"],      # Rakhine (Myanmar)
    "shn": ["Asia"],      # Shan (Myanmar)
    "syl": ["Asia"],      # Sylheti (Bangladesh/India)
    "szy": ["Asia"],      # Sakizaya (Taiwan)
    "tcy": ["Asia"],      # Tulu (India)
    "tdd": ["Asia"],      # Tai Nüa (China/Myanmar)
    "trv": ["Asia"],      # Sediq (Taiwan)
    "za": ["Asia"],       # Zhuang (China)

    # Europe
    "be-tarask": ["Europe"],   # Belarusian (Taraskievica)
    "frp": ["Europe"],    # Arpitan/Franco-Provençal
    "gag": ["Europe"],    # Gagauz (Moldova)
    "got": ["Europe"],    # Gothic (ancient)
    "lld": ["Europe"],    # Ladin (Italy)
    "nds-nl": ["Europe"], # Low Saxon (Netherlands)
    "rmy": ["Europe"],    # Vlax Romani
    "roa-tara": ["Europe"],  # Tarantino (Italy)
    "rsk": ["Europe"],    # Ruthenian
    "rup": ["Europe"],    # Macedo-Romanian
    "sco": ["Europe"],    # Scots (Scotland)
    "sh": ["Europe"],     # Serbo-Croatian

    # North America
    "cho": ["North America"],  # Choctaw

    # Constructed/International
    "cbk-zam": ["Asia"],  # Chavacano (Philippines)
    "lfn": ["Europe"],    # Lingua Franca Nova (constructed)
    "map-bms": ["Asia"],  # Banyumasan (Indonesia)
    "simple": ["Europe", "North America"],  # Simple English
    "tok": ["Europe"],    # Toki Pona (constructed)
}

async def main():
    # Load current languages
    data_path = Path(__file__).parent.parent / "src" / "data"
    languages_path = data_path / "languages.json"

    with open(languages_path) as f:
        languages = json.load(f)

    print(f"Processing {len(languages)} languages...")

    # Build language-to-continents mapping
    lang_continents = {}

    for lang in languages:
        code = lang['code']

        # Use known mappings first
        if code in KNOWN_LANGUAGE_CONTINENTS:
            lang_continents[code] = KNOWN_LANGUAGE_CONTINENTS[code]
        else:
            # Default to empty (will be "Worldwide" or uncategorized)
            lang_continents[code] = []

    # Count stats
    mapped = sum(1 for c in lang_continents.values() if c)
    unmapped = len(languages) - mapped

    print(f"  Mapped: {mapped}")
    print(f"  Unmapped: {unmapped}")

    if unmapped > 0:
        print(f"\n  Unmapped languages:")
        for lang in languages:
            if not lang_continents.get(lang['code']):
                print(f"    - {lang['code']}: {lang.get('common_name') or lang['name']}")

    # Write output
    output_path = data_path / "language_continents.json"
    with open(output_path, 'w') as f:
        json.dump(lang_continents, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(lang_continents)} mappings to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
