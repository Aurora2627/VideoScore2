"""Leakage, objective, metric and CUDA parameter-update checks."""
import copy,unittest
import numpy as np
import torch
from prepare_data import partition,canonical,LABELS
from train_heads import ScoreHead,ranking_loss,normalize_train,evaluate
class TrainingTests(unittest.TestCase):
 def fixture(self):
  train=[{'video_name':str(i).zfill(6)+'_'+c,'prompt':'prompt '+str(i),'prompt_id':str(i).zfill(6),'ground_truth':dict.fromkeys(LABELS,3),'source_split':'official_train'} for i in range(12) for c in ('a','b')]
  bench=[{'video_name':'000099_'+c,'prompt':'unseen video','visual_score':3,'t2v_score':3,'phy_score':3,'video_url':'https://example.org/'+c} for c in ('a','b')]
  return train,bench
 def test_prompt_disjoint_and_benchmark_exclusion(self):
  train,bench=self.fixture();train.append(dict(train[0],video_name='000099_z',prompt_id='000099',prompt='unseen video'))
  splits=partition(train,bench,set(),4,2,1,2)
  for a,b in [('train','validation'),('train','test'),('validation','test')]:self.assertFalse({canonical(r['prompt']) for r in splits[a]} & {canonical(r['prompt']) for r in splits[b]})
  self.assertTrue(all(r['source_split']=='official_train' for r in splits['train']+splits['validation']))
 def test_split_independent_of_labels(self):
  train,bench=self.fixture();before=partition(train,bench,set(),4,2,1,2)
  for r in train:r['ground_truth']=dict.fromkeys(LABELS,1)
  for r in bench:r.update(visual_score=5,t2v_score=1,phy_score=5)
  after=partition(train,bench,set(),4,2,1,2)
  self.assertEqual({k:[r['video_name'] for r in v] for k,v in before.items()},{k:[r['video_name'] for r in v] for k,v in after.items()})
 def test_previously_seen_prompt_not_test(self):
  train,bench=self.fixture()
  with self.assertRaises(AssertionError):partition(train,bench,{'000099_a'},4,2,1,2)
 def test_duplicate_prompt_across_ids_stays_together(self):
  train,bench=self.fixture();train[2]['prompt']=train[0]['prompt'].upper()+'.'
  splits=partition(train,bench,set(),4,2,1,2);owners={}
  for split,rows in splits.items():
   for r in rows:
    key=canonical(r['prompt']);self.assertIn(owners.get(key,split),(split,));owners[key]=split
 def test_ranking_objective_and_no_cross_prompt_pairs(self):
  y=torch.tensor([[5.,4.,3.],[1.,2.,1.]]);good=y.clone().requires_grad_();bad=y.flip(0).clone().requires_grad_()
  self.assertLess(float(ranking_loss(good,y,['same','same'])),float(ranking_loss(bad,y,['same','same'])))
  zero=ranking_loss(bad,y,['one','two']);self.assertEqual(float(zero),0.);zero.backward();self.assertTrue(torch.equal(bad.grad,torch.zeros_like(bad)))
 def test_annotation_tie_excluded_and_prediction_tie_wrong(self):
  rows=[{'prompt':'same','video_name':str(i)} for i in range(3)];target=np.array([[1.,1.,1.],[5.,5.,5.],[5.,5.,5.]])
  result=evaluate(np.full((3,3),3.),target,rows);self.assertEqual(result['ranking_pairs'],2);self.assertEqual(result['annotation_ties_excluded'],1);self.assertEqual(result['ranking_correct'],0)
 def test_normalization_finite_for_constant_dimensions(self):
  x=torch.tensor([[1.,2.],[1.,4.]]);mean,std=normalize_train(x);self.assertTrue(torch.isfinite((x-mean)/std).all());self.assertTrue(torch.equal(mean,torch.tensor([1.,3.])))
 @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for real update smoke')
 def test_cuda_head_really_learns_and_checkpoint_restores(self):
  torch.manual_seed(42);model=ScoreHead(8,'linear').cuda();x=torch.randn(32,8,device='cuda');y=1+4*torch.sigmoid(x[:,:3]);opt=torch.optim.AdamW(model.parameters(),lr=.03)
  initial=float((model(x)-y).square().mean())
  for _ in range(160):
   loss=(model(x)-y).square().mean();opt.zero_grad();loss.backward();opt.step()
  model.eval();p=model(x).detach();self.assertLess(float((p-y).square().mean()),initial*.1);self.assertTrue(bool(((p>=1)&(p<=5)).all()))
  restored=ScoreHead(8,'linear').cuda();restored.load_state_dict(model.state_dict());restored.eval();self.assertTrue(torch.equal(p,restored(x).detach()))
if __name__=='__main__':unittest.main()
