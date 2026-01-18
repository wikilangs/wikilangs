
import sys
import subprocess
import pytest

def run_import_test(mock_modules):
    """Run an import test in a subprocess with mocked modules."""
    mock_code = ""
    for module in mock_modules:
        mock_code += f"import sys; sys.modules['{module}'] = None; "
    
    code = f"{mock_code} import wikilangs; import wikilangs.tokenizer; print('SUCCESS')"
    
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True
    )
    return result

def test_tokenizer_import_without_transformers():
    """Test that tokenizer can be imported even without transformers."""
    result = run_import_test(["transformers", "tokenizers"])
    assert result.returncode == 0, f"Import failed with error: {result.stderr}"
    assert "SUCCESS" in result.stdout

def test_embeddings_import_without_babelvec():
    """Test that embeddings can be imported even without babelvec."""
    result = run_import_test(["babelvec"])
    assert result.returncode == 0, f"Import failed with error: {result.stderr}"
    assert "SUCCESS" in result.stdout

def test_llm_import_without_torch_transformers():
    """Test that llm utilities are handled gracefully when dependencies are missing."""
    # Note: llm is imported in __init__.py but inside a try-except.
    # We want to make sure the package itself still imports.
    code = "import sys; sys.modules['transformers'] = None; sys.modules['torch'] = None; import wikilangs; print('SUCCESS')"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Package import failed with error: {result.stderr}"
    assert "SUCCESS" in result.stdout
