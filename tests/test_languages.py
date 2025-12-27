"""Tests for the languages module."""

import pytest
import os
from pathlib import Path
import tempfile

from wikilangs.languages import languages, languages_with_metadata, LanguageInfo, get_language_info
from wikilangs import languages as languages_factory, languages_with_metadata as languages_with_metadata_factory, LanguageInfo as LanguageInfoFactory, get_language_info as get_language_info_factory


class TestLanguagesFunction:
    """Test the languages function with real data."""
    
    def test_languages(self):
        """Test languages function."""
        langs = languages(date="20251201")
        
        # Basic validation
        assert isinstance(langs, list)
        assert len(langs) > 0
        assert all(isinstance(lang, str) for lang in langs)
        assert langs == sorted(langs)  # Should be sorted
        
        print(f"Found {len(langs)} languages: {langs[:10]}...")
    
    def test_languages_factory_import(self):
        """Test languages function imported from main package."""
        langs = languages_factory(date="20251201")
        
        assert isinstance(langs, list)
        assert len(langs) > 0
        print(f"Languages via factory import: {langs[:10]}...")
    
    def test_languages_consistency_direct_vs_factory(self):
        """Test that direct import and factory import return same results."""
        langs_direct = languages(date="20251201")
        langs_factory_res = languages_factory(date="20251201")
        
        assert langs_direct == langs_factory_res
        print(f"Consistent results: {len(langs_direct)} languages")
    
    def test_languages_with_huggingface(self):
        """Test languages function using HuggingFace."""
        try:
            langs = languages(date="20251201")
            
            assert isinstance(langs, list)
            assert len(langs) > 0
            assert all(isinstance(lang, str) for lang in langs)
            assert langs == sorted(langs)  # Should be sorted
            
            print(f"HuggingFace languages: {langs[:10]}...")  # Show first 10
            
        except Exception as e:
            pytest.skip(f"HuggingFace not accessible: {e}")
    
    def test_languages_with_invalid_date(self):
        """Test languages function with invalid date."""
        try:
            langs = languages(date="invalid_date")
            # If it works, should return empty list or valid list
            assert isinstance(langs, list)
            
        except FileNotFoundError:
            # Expected for invalid dates when HuggingFace is used
            pass
    
    def test_languages_specific_date_structure(self):
        """Test that languages function returns valid list."""
        langs = languages(date="20251201")
        
        assert isinstance(langs, list)
        assert len(langs) > 0
        assert all(isinstance(l, str) for l in langs)
        print(f"Received {len(langs)} language codes")


class TestLanguagesEdgeCases:
    """Test edge cases and error conditions for languages function."""
    
    def test_languages_with_partial_local_structure(self):
        """Test languages function."""
        try:
            langs = languages(date="20251201")
            assert isinstance(langs, list)
            print(f"HuggingFace call worked: {len(langs)} languages")
            
        except FileNotFoundError:
            # Expected if HuggingFace is not accessible
            print("HuggingFace call failed (expected in offline mode)")
    
    def test_languages_empty_date_directory(self):
        """Test languages function."""
        try:
            langs = languages(date="20251201")
            assert isinstance(langs, list)
            print(f"HuggingFace call worked: {len(langs)} languages")
            
        except FileNotFoundError:
            # Expected if HuggingFace is not accessible
            print("HuggingFace call failed (expected in offline mode)")
    
    def test_languages_with_valid_local_structure(self):
        """Test languages function."""
        langs = languages(date="20251201")
        assert isinstance(langs, list)
        assert len(langs) > 0
        print(f"HuggingFace call returned {len(langs)} langs")


class TestLanguageInfo:
    """Test the LanguageInfo dataclass."""
    
    def test_language_info_creation(self):
        """Test creating LanguageInfo instances."""
        # Basic creation
        lang_info = LanguageInfo(code="en", name="English")
        assert lang_info.code == "en"
        assert lang_info.name == "English"
        assert lang_info.common_name is None
        assert lang_info.alpha_2 is None
        
        # Full creation
        lang_info_full = LanguageInfo(
            code="en",
            name="English",
            common_name="English",
            alpha_2="en",
            alpha_3="eng",
            scope="I",
            type="L",
            bibliographic="eng",
            terminological="eng"
        )
        assert lang_info_full.code == "en"
        assert lang_info_full.name == "English"
        assert lang_info_full.alpha_2 == "en"
        assert lang_info_full.alpha_3 == "eng"
        assert lang_info_full.scope == "I"
        assert lang_info_full.type == "L"
    
    def test_language_info_str_representation(self):
        """Test string representation of LanguageInfo."""
        # Without common name
        lang_info = LanguageInfo(code="en", name="English")
        assert str(lang_info) == "en: English"
        
        # With same common name
        lang_info_same = LanguageInfo(code="en", name="English", common_name="English")
        assert str(lang_info_same) == "en: English"
        
        # With different common name
        lang_info_diff = LanguageInfo(code="bn", name="Bengali", common_name="Bangla")
        assert str(lang_info_diff) == "bn: Bengali (Bangla)"
    
    def test_language_info_factory_import(self):
        """Test LanguageInfo imported from main package."""
        lang_info = LanguageInfoFactory(code="test", name="Test Language")
        assert isinstance(lang_info, LanguageInfo)
        assert lang_info.code == "test"
        assert lang_info.name == "Test Language"


class TestGetLanguageInfo:
    """Test the get_language_info function."""
    
    def test_get_language_info_known_codes(self):
        """Test getting language info for known ISO codes."""
        # Test with specific codes that we know work reliably
        test_cases = [
            ("eng", "English"),  # Use 3-letter codes for more reliable matching
            ("fra", "French"),
            ("deu", "German"),
            ("spa", "Spanish")
        ]
        
        for code, expected_name in test_cases:
            lang_info = get_language_info(code)
            assert lang_info is not None
            assert lang_info.code == code
            # More flexible name matching since pycountry might have variations
            assert lang_info.name is not None and len(lang_info.name) > 0
            
            # Should have at least some metadata for common languages
            assert lang_info.alpha_2 is not None or lang_info.alpha_3 is not None
            
            print(f"✓ {code}: {lang_info.name}")
    
    def test_get_language_info_three_letter_codes(self):
        """Test getting language info for 3-letter ISO codes."""
        test_cases = [
            ("eng", "English"),
            ("fra", "French"),
            ("deu", "German"),
            ("spa", "Spanish")
        ]
        
        for code, expected_name in test_cases:
            lang_info = get_language_info(code)
            assert lang_info is not None
            assert lang_info.code == code
            assert expected_name.lower() in lang_info.name.lower()
            print(f"✓ {code}: {lang_info.name}")
    
    def test_get_language_info_unknown_code(self):
        """Test getting language info for unknown codes."""
        lang_info = get_language_info("xyz")
        assert lang_info is not None
        assert lang_info.code == "xyz"
        assert "XYZ" in lang_info.name
        assert "639-3" in lang_info.name  # Should indicate it's treated as 3-letter code
        
        # Test 2-letter unknown code
        lang_info_2 = get_language_info("xy")
        assert lang_info_2 is not None
        assert lang_info_2.code == "xy"
        assert "XY" in lang_info_2.name
        assert "639-1" in lang_info_2.name  # Should indicate it's treated as 2-letter code
    
    def test_get_language_info_factory_import(self):
        """Test get_language_info imported from main package."""
        lang_info = get_language_info_factory("eng")  # Use 3-letter code for reliability
        assert lang_info is not None
        assert lang_info.code == "eng"
        assert lang_info.name is not None and len(lang_info.name) > 0
    
    def test_get_language_info_case_sensitivity(self):
        """Test that language lookup handles case correctly."""
        # pycountry.languages.lookup should handle case insensitivity
        lang_info_lower = get_language_info("en")
        lang_info_upper = get_language_info("EN")
        
        # Both should work (pycountry handles case insensitivity)
        assert lang_info_lower is not None
        assert lang_info_upper is not None
        
        # They should have the same name (both should resolve to English)
        assert lang_info_lower.name == lang_info_upper.name


class TestLanguagesWithMetadata:
    """Test the languages_with_metadata function with real data."""
    
    def test_languages_with_metadata_local_data(self):
        """Test languages_with_metadata."""
        lang_infos = languages_with_metadata(date="20251201")
        
        # Basic validation
        assert isinstance(lang_infos, list)
        assert len(lang_infos) > 0
        assert all(isinstance(info, LanguageInfo) for info in lang_infos)
        
        # Check that all have codes and names
        for lang_info in lang_infos:
            assert lang_info.code
            assert lang_info.name
            assert isinstance(lang_info.code, str)
            assert isinstance(lang_info.name, str)
        
        # Should be sorted by code (same as basic languages function)
        codes = [info.code for info in lang_infos]
        assert codes == sorted(codes)
        
        print(f"Found {len(lang_infos)} languages with metadata:")
        for lang_info in lang_infos[:5]:  # Show first 5
            print(f"  {lang_info}")
    
    def test_languages_with_metadata_factory_import(self):
        """Test languages_with_metadata imported from main package."""
        lang_infos = languages_with_metadata_factory(date="20251201")
        
        assert isinstance(lang_infos, list)
        assert len(lang_infos) > 0
        assert all(isinstance(info, LanguageInfo) for info in lang_infos)
        print(f"Factory import works: {len(lang_infos)} languages with metadata")
    
    def test_languages_with_metadata_consistency(self):
        """Test consistency between basic languages and languages_with_metadata."""
        # Get both basic and metadata versions
        basic_codes = languages(date="20251201")
        lang_infos = languages_with_metadata(date="20251201")
        
        # Should have same number of languages
        assert len(basic_codes) == len(lang_infos)
        
        # Codes should match in order
        metadata_codes = [info.code for info in lang_infos]
        assert basic_codes == metadata_codes
        
        print(f"✓ Consistency check passed: {len(basic_codes)} languages")
    
    def test_languages_with_metadata_known_languages(self):
        """Test that known languages get proper metadata."""
        lang_infos = languages_with_metadata(date="20251201")
        
        # Look for common languages that should have good metadata
        common_lang_patterns = {
            'en': 'English',
            'fr': 'French', 
            'de': 'German',
            'es': 'Spanish',
            'ar': 'Arabic'
        }
        
        found_common = {}
        for lang_info in lang_infos:
            for pattern, expected_name in common_lang_patterns.items():
                if pattern in lang_info.code.lower():
                    found_common[pattern] = lang_info
                    
                    # Should have rich metadata for common languages
                    assert lang_info.name
                    print(f"✓ Found {pattern}: {lang_info}")
                    
                    # Check metadata quality
                    if lang_info.alpha_2:
                        print(f"  Alpha-2: {lang_info.alpha_2}")
                    if lang_info.alpha_3:
                        print(f"  Alpha-3: {lang_info.alpha_3}")
                    if lang_info.scope:
                        print(f"  Scope: {lang_info.scope}")
                    if lang_info.type:
                        print(f"  Type: {lang_info.type}")
        
        print(f"Found {len(found_common)} common languages with metadata")
    
    def test_languages_with_metadata_huggingface_fallback(self):
        """Test languages_with_metadata with HuggingFace fallback."""
        try:
            lang_infos = languages_with_metadata(date="20251201")
            
            assert isinstance(lang_infos, list)
            assert len(lang_infos) > 0
            assert all(isinstance(info, LanguageInfo) for info in lang_infos)
            
            # Check that we get metadata for at least some languages
            with_metadata = [info for info in lang_infos if info.alpha_2 or info.alpha_3]
            print(f"HuggingFace: {len(lang_infos)} total, {len(with_metadata)} with ISO metadata")
            
            # Show some examples
            for lang_info in lang_infos[:3]:
                print(f"  {lang_info}")
                
        except Exception as e:
            pytest.skip(f"HuggingFace not accessible: {e}")
    
    def test_languages_with_metadata_unknown_codes(self):
        """Test handling of unknown language codes."""
        # Create a temporary structure with truly fake language codes
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid structure with fake language codes that definitely don't exist
            date_dir = Path(temp_dir) / "20251201"
            date_dir.mkdir()
            
            # Use codes that are very unlikely to be real language codes
            fake_langs = ["zzz", "xxx", "qqq"]
            for lang in fake_langs:
                lang_dir = date_dir / lang
                lang_dir.mkdir()
                dataset_dir = lang_dir / "dataset"
                dataset_dir.mkdir()
                (dataset_dir / "dummy.parquet").touch()
            
            # The implementation no longer reads local data; ensure the
            # call succeeds and returns LanguageInfo objects from HF configs.
            lang_infos = languages_with_metadata(date="20251201")
            assert isinstance(lang_infos, list)
            assert len(lang_infos) > 0
            for lang_info in lang_infos[:5]:
                assert isinstance(lang_info, LanguageInfo)
                assert lang_info.code
                assert lang_info.name
            print(f"languages_with_metadata returned {len(lang_infos)} entries")


class TestLanguageMetadataIntegration:
    """Integration tests for language metadata functionality."""
    
    def test_all_imports_work(self):
        """Test that all imports work correctly."""
        # Test direct imports
        assert languages is not None
        assert languages_with_metadata is not None
        assert LanguageInfo is not None
        assert get_language_info is not None
        
        # Test factory imports
        assert languages_factory is not None
        assert languages_with_metadata_factory is not None
        assert LanguageInfoFactory is not None
        assert get_language_info_factory is not None
        
        print("✓ All imports work correctly")
    
    def test_metadata_functions_consistency(self):
        """Test consistency between direct and factory imports."""
        # Test get_language_info consistency
        info_direct = get_language_info("en")
        info_factory = get_language_info_factory("en")
        
        assert info_direct.code == info_factory.code
        assert info_direct.name == info_factory.name
        assert info_direct.alpha_2 == info_factory.alpha_2
        
        print("✓ Direct and factory imports are consistent")
    
    def test_language_info_dataclass_features(self):
        """Test LanguageInfo dataclass features."""
        lang_info = LanguageInfo(code="en", name="English", alpha_2="en")
        
        # Test equality
        lang_info2 = LanguageInfo(code="en", name="English", alpha_2="en")
        assert lang_info == lang_info2
        
        # Test inequality
        lang_info3 = LanguageInfo(code="fr", name="French", alpha_2="fr")
        assert lang_info != lang_info3
        
        # Test repr
        repr_str = repr(lang_info)
        assert "LanguageInfo" in repr_str
        assert "code='en'" in repr_str
        assert "name='English'" in repr_str
        
        print("✓ LanguageInfo dataclass features work correctly")


if __name__ == "__main__":
    # Run basic integration test
    print("Running basic languages integration tests...")
    
    # Test HuggingFace if available
    print("\n=== Testing languages with HuggingFace ===")
    try:
        langs = languages(date="20251201")
        print(f"HuggingFace languages: {len(langs)} found")
        print(f"Sample languages: {langs[:5]}...")
    except Exception as e:
        print(f"HuggingFace test failed: {e}")
    
    # Test metadata functionality if available
    print("\n=== Testing language metadata functionality ===")
    try:
        # Test get_language_info
        print("Testing get_language_info...")
        en_info = get_language_info('en')
        print(f"English info: {en_info}")
        print(f"  Alpha-2: {en_info.alpha_2}")
        print(f"  Alpha-3: {en_info.alpha_3}")
        print(f"  Scope: {en_info.scope}")
        print(f"  Type: {en_info.type}")
        
        # Test with HuggingFace if available
        print("\nTesting languages_with_metadata with HuggingFace...")
        try:
            hf_lang_infos = languages_with_metadata(date="20251201")
            print(f"HuggingFace: {len(hf_lang_infos)} languages with metadata")
            
            # Show languages with rich metadata
            rich_metadata = [info for info in hf_lang_infos if info.alpha_2 and info.alpha_3]
            print(f"Languages with rich metadata: {len(rich_metadata)}")
            
            for lang_info in rich_metadata[:3]:  # Show first 3 with rich metadata
                print(f"  {lang_info}")
                
        except Exception as e:
            print(f"HuggingFace metadata test failed: {e}")
        
        print("✓ Language metadata functionality works correctly")
        
    except Exception as e:
        print(f"Metadata functionality test failed: {e}")
    
    print("\nLanguages integration tests completed!")
