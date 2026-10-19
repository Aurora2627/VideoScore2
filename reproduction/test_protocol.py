"""Fast tests for failure-sensitive parsing; no model metrics."""
import unittest
from infer_cuda import parse_scores, official_prompt
class ProtocolTests(unittest.TestCase):
    text='visual quality: 3; text-to-video alignment: 4; physical/common-sense consistency: 5'
    def test_valid(self): self.assertTrue(parse_scores(self.text)['valid'])
    def test_missing(self): self.assertFalse(parse_scores('visual quality: 3')['valid'])
    def test_range(self): self.assertFalse(parse_scores(self.text.replace('quality: 3','quality: 10'))['valid'])
    def test_duplicate(self): self.assertFalse(parse_scores(self.text+'; visual quality: 1')['valid'])
    def test_decimal(self): self.assertFalse(parse_scores(self.text.replace('quality: 3','quality: 3.5'))['valid'])
    def test_official_prompt(self): self.assertIn('sample generation prompt',official_prompt('sample generation prompt'))
if __name__=='__main__': unittest.main()
