import json,unittest
from pairwise import parse_comparison,order_consensus,KEYS,LABELS
WEIGHTS=dict(zip(LABELS,(.3,.4,.3)))
def raw(winner='A'):return '<comparison>'+json.dumps({key:{'winner':winner,'evidence':'A has sharper detail than B.'} for key in KEYS})+'</comparison>'
def parsed(winner):return parse_comparison(raw(winner),True)
class PairwiseTests(unittest.TestCase):
    def test_final_block_is_parsed_after_reasoning(self):self.assertTrue(parse_comparison('<think>example '+raw('B')+'</think>'+raw('A'),True)['valid'])
    def test_truncation_is_invalid(self):self.assertFalse(parse_comparison(raw(),False)['valid'])
    def test_multiple_final_blocks_are_invalid(self):self.assertFalse(parse_comparison(raw()+raw(),True)['valid'])
    def test_swapped_consistent_winner_maps_to_same_video(self):self.assertEqual(order_consensus(parsed('A'),parsed('B'),['clip1','clip2'],WEIGHTS)['selected'],'clip1')
    def test_same_display_position_is_rejected(self):self.assertEqual(order_consensus(parsed('A'),parsed('A'),['clip1','clip2'],WEIGHTS)['reason'],'order_disagreement')
    def test_uncertain_abstains(self):self.assertIsNone(order_consensus(parsed('uncertain'),parsed('uncertain'),['clip1','clip2'],WEIGHTS)['selected'])
    def test_tie_does_not_get_a_forced_winner(self):self.assertEqual(order_consensus(parsed('tie'),parsed('tie'),['clip1','clip2'],WEIGHTS)['reason'],'weighted_tie')
    def test_short_evidence_is_invalid(self):self.assertFalse(parse_comparison(raw().replace('A has sharper detail than B.','ok'),True)['valid'])
if __name__=='__main__':unittest.main()
