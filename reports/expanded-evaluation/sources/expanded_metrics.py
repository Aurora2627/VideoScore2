"""Paired evaluation with prompt-level uncertainty; invalid outputs remain visible."""
import collections,random,statistics
from quality_metrics import LABELS,evaluate,valid

def prompt_bootstrap(values, draws=5000, seed=20261022):
    if not values: return None
    rng=random.Random(seed);n=len(values)
    means=sorted(statistics.fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(draws))
    return {'mean':statistics.fmean(values),'lower_95':means[int(.025*(draws-1))],'upper_95':means[int(.975*(draws-1))],
            'prompt_groups':n,'draws':draws,'seed':seed,'direction':'detail minus baseline; negative means lower error'}

def correlations(samples,results):
    from scipy.stats import spearmanr,pearsonr
    output={}
    for key in LABELS:
        usable=[item for item in samples if valid(results.get(item['video_name'],{}))]
        truth=[float(item['ground_truth'][key]) for item in usable]
        predicted=[float(results[item['video_name']]['scores'][key]) for item in usable]
        usable_variation=len(truth)>=3 and len(set(truth))>1 and len(set(predicted))>1
        output[key]={'count':len(truth),'spearman':float(spearmanr(truth,predicted).statistic) if usable_variation else None,
                     'pearson':float(pearsonr(truth,predicted).statistic) if usable_variation else None}
    return output

def summarize(samples, baseline, detail, weights):
    paired=[item for item in samples if valid(baseline.get(item['video_name'],{})) and valid(detail.get(item['video_name'],{}))]
    groups=collections.defaultdict(list)
    for item in samples: groups[item['prompt']].append(item)
    complete_groups=[items for items in groups.values() if all(item in paired for item in items)]
    differences=[]
    for items in complete_groups:
        changes=[]
        for item in items:
            name=item['video_name'];truth=item['ground_truth']
            changes.append(statistics.fmean(abs(detail[name]['scores'][key]-float(truth[key]))-abs(baseline[name]['scores'][key]-float(truth[key])) for key in LABELS))
        differences.append(statistics.fmean(changes))
    full=len(paired)==len(samples)
    bm=evaluate(samples,baseline,weights);dm=evaluate(samples,detail,weights)
    paired_b=evaluate(paired,baseline,weights);paired_d=evaluate(paired,detail,weights)
    constant={item['video_name']:{'status':'completed','valid':True,'eos_reached':True,'scores':{key:3 for key in LABELS}} for item in samples}
    resource={}
    for name,results in [('baseline',baseline),('more_detail',detail)]:
        values=list(results.values());times=[r['generation_seconds'] for r in values if 'generation_seconds' in r]
        resource[name]={'attempted':len(values),'mean_generation_seconds':statistics.fmean(times) if times else None,
                        'peak_allocated_gib':max((r.get('peak_allocated_bytes',0) for r in values),default=0)/2**30}
    return {'planned_videos':len(samples),'planned_prompt_groups':len(groups),'paired_valid_videos':len(paired),
            'complete_prompt_groups':len(complete_groups),'full_coverage':full,'baseline':bm,'more_detail':dm,
            'correlations':{'baseline':correlations(paired,baseline),'more_detail':correlations(paired,detail)},'paired_baseline':paired_b,'paired_more_detail':paired_d,'constant_three_reference':evaluate(samples,constant,weights),
            'paired_prompt_bootstrap':prompt_bootstrap(differences),'resources':resource,
            'conclusion':'incomplete_coverage' if not full else ('lower_observed_mae' if dm['macro_mae']<bm['macro_mae'] else 'no_observed_mae_improvement'),
            'scope':'20 fresh videos from 10 prompts is a diagnostic subset. No parameter fitting, online effect or full-benchmark claim.'}
