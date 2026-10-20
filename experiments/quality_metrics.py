"""Predeclared diagnostic metrics and conservative configuration selection."""
import itertools,math
LABELS=('visual quality','text-to-video alignment','physical/common-sense consistency')
def valid(result):
    scores=result.get('scores') or {}
    return result.get('status')=='completed' and result.get('valid') is True and result.get('eos_reached') is True and set(scores)==set(LABELS) and all(type(v)==int and 1<=v<=5 for v in scores.values())
def evaluate(samples,results,weights):
    errors={k:[] for k in LABELS};groups={};passed=[]
    for item in samples:
        name=item['video_name'];groups.setdefault(item['prompt'],[]).append(item)
        if not valid(results.get(name,{})):continue
        passed.append(name)
        for k in LABELS:errors[k].append(abs(results[name]['scores'][k]-float(item['ground_truth'][k])))
    mae={k:sum(v)/len(v) if v else None for k,v in errors.items()};correct=total=0
    for group in groups.values():
        for a,b in itertools.combinations(group,2):
            if a['video_name'] not in passed or b['video_name'] not in passed:continue
            margin=sum((float(a['ground_truth'][k])-float(b['ground_truth'][k]))*weights[k] for k in LABELS)
            if abs(margin)<1e-8:continue
            predicted=sum((results[a['video_name']]['scores'][k]-results[b['video_name']]['scores'][k])*weights[k] for k in LABELS)
            total+=1;correct+=int(margin*predicted>1e-8)
    return {'sample_count':len(samples),'valid_count':len(passed),'valid_rate':len(passed)/len(samples) if samples else 0,'per_dimension_mae':mae,'macro_mae':sum(mae.values())/3 if passed else None,'pair_correct':correct,'pair_total':total,'pair_agreement':correct/total if total else None}
def choose(baseline,candidates,configurations):
    eligible=[name for name,metrics in candidates.items() if metrics['valid_rate']==1 and metrics['macro_mae'] is not None and metrics['macro_mae']<baseline['macro_mae']-1e-8]
    return min(eligible,key=lambda name:(candidates[name]['macro_mae'],configurations[name]['frames']*configurations[name]['max_side']**2)) if eligible else 'baseline'
def promotion_gate(dev,regression_base,regression,check_base,check):
    return {'all_outputs_complete':all(m['valid_rate']==1 for m in (dev,regression_base,regression,check_base,check)), 'development_pair_correct':dev['pair_total']>0 and dev['pair_correct']==dev['pair_total'], 'regression_mae_not_worse':regression['macro_mae'] is not None and regression['macro_mae']<=regression_base['macro_mae']+1e-8, 'fresh_check_mae_improved':check['macro_mae'] is not None and check_base['macro_mae'] is not None and check['macro_mae']<check_base['macro_mae']-1e-8, 'fresh_check_pair_correct':check['pair_total']>0 and check['pair_correct']==check['pair_total']}
