"""Transparent, prompt-grouped selection policy; scores are not probabilities."""
import hashlib,math
LABELS=('visual quality','text-to-video alignment','physical/common-sense consistency')
DEFAULT={'weights':dict(zip(LABELS,(0.3,0.4,0.3))),'minimum_scores':dict(zip(LABELS,(3,3,3))),'top_k':1,'policy_status':'heuristic_not_calibrated'}

def validate_policy(policy):
    weights=policy['weights'];minimum=policy['minimum_scores']
    if set(weights)!=set(LABELS) or set(minimum)!=set(LABELS):raise ValueError('All three score dimensions are required')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in weights.values()) or abs(sum(weights.values())-1)>1e-8:raise ValueError('Weights must be finite, nonnegative and sum to one')
    if any(type(v)!=int or not 1<=v<=5 for v in minimum.values()):raise ValueError('Minimum scores must be integers from 1 to 5')
    if type(policy['top_k'])!=int or policy['top_k']<1:raise ValueError('top_k must be positive')

def select_candidates(samples,results,policy):
    validate_policy(policy);groups={}
    for sample in samples:
        prompt=sample['prompt'];key=hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        group=groups.setdefault(key,{'prompt':prompt,'prompt_sha256':key,'candidates':[]})
        result=results.get(sample['video_name'],{});scores=result.get('scores',{})
        valid=result.get('status')=='completed' and result.get('eos_reached') is True and result.get('valid') is True and isinstance(scores,dict) and set(scores)==set(LABELS) and all(type(v)==int and 1<=v<=5 for v in scores.values())
        row={'pending':sample['video_name'] not in results,'video_name':sample['video_name'],'scores':scores,'rank':None,'selected':False,'valid':valid}
        if valid:
            row['weighted_score']=round(sum(scores[k]*policy['weights'][k] for k in LABELS),6)
            row['review_reasons']=[k+' below minimum' for k in LABELS if scores[k]<policy['minimum_scores'][k]]
        else:row.update(weighted_score=None,review_reasons=['Incomplete or invalid model output'])
        group['candidates'].append(row)
    for group in groups.values():
        processing=any(r['pending'] for r in group['candidates'])
        group['processing']=processing
        comparable=not processing and sum(r['valid'] for r in group['candidates'])>=2
        group['comparison_available']=comparable
        eligible=sorted((r for r in group['candidates'] if r['valid'] and not r['review_reasons']),key=lambda r:(-r['weighted_score'],r['video_name']))
        previous=None;rank=0
        for position,row in enumerate(eligible,1):
            if row['weighted_score']!=previous:rank=position;previous=row['weighted_score']
            if comparable:row['rank']=rank;row['selected']=rank<=policy['top_k']
        for row in group['candidates']:
            row['decision']='pending' if row['pending'] else ('comparison_pending' if processing and row['valid'] and not row['review_reasons'] else ('manual_review' if row['review_reasons'] else ('single_candidate_review' if not comparable else ('shortlist' if row['selected'] else 'reserve'))))
        group['note']='Ties share rank and may expand the shortlist. Shortlists require human approval.'
    return {'policy':policy,'groups':list(groups.values()),'automatic_publication':False}
