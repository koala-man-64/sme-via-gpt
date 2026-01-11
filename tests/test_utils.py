
import unittest
from app import clean_text, chunk_by_tokens, _format_env_value

class TestUtils(unittest.TestCase):
    def test_clean_text(self):
        # Test basic cleanup
        self.assertEqual(clean_text("  hello   world  "), "hello world")
        # Test null bytes
        self.assertEqual(clean_text("hello\x00world"), "hello world")
        # Test consecutive newlines
        self.assertEqual(clean_text("hello\n\n\nworld"), "hello\n\nworld")

    def test_format_env_value(self):
        self.assertEqual(_format_env_value("ANY_KEY", None), "<unset>")
        self.assertEqual(_format_env_value("ANY_KEY", ""), "<empty>")
        self.assertEqual(_format_env_value("OPENAI_API_KEY", "sk-1234567890"), "****7890")
        self.assertEqual(_format_env_value("OTHER_KEY", "sensitive"), "sensitive")

    def test_chunk_by_tokens(self):
        text = "hello world " * 50
        # Mocking tiktoken inside the app is hard without refactoring, 
        # but the function uses tiktoken.get_encoding.
        # We assume standard encoding exists or we can mock it if needed.
        # For this integration level test, we'll let it use the real tokenizer 
        # as it's a library call.
        
        chunks = chunk_by_tokens(text, chunk_tokens=10, overlap=0)
        self.assertTrue(len(chunks) > 1)
        
        # Test overlap
        chunks_overlap = chunk_by_tokens(text, chunk_tokens=10, overlap=5)
        self.assertTrue(len(chunks_overlap) > len(chunks))

if __name__ == "__main__":
    unittest.main()
