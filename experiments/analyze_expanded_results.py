"""Inspect distribution of gains and remaining ranking errors in the frozen evaluation."""
import collections,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LABELS=('visual quality','text-to-video alignment','physical/common-sense consistency')
def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def main():
    report=ROOT/'reports/expanded-evaluation';data=read(report/'manifest.json');weights=read(report/'policy.json')['weights']
    summary=read(report/'summary.json');assert summary['full_coverage'];groups=collections.defaultdict(list);histograms={};results={}
    for phase in ('baseline','more_detail'):
        results[phase]={item['video_name']:read(report/phase/item['video_name']/'result.json') for item in data['samples']}
        histograms[phase]={key:dict(collections.Counter(r['scores'][key] for r in results[phase].values())) for key in LABELS}
    for item in data['samples']:groups[item['prompt']].append(item)
    per_group=[];rows=[]
    for prompt,items in groups.items():
        errors={}
        for phase in results:
            errors[phase]=statistics.fmean(abs(results[phase][item['video_name']]['scores'][key]-float(item['ground_truth'][key])) for item in items for key in LABELS)
        truth_margin=sum((float(items[0]['ground_truth'][key])-float(items[1]['ground_truth'][key]))*weights[key] for key in LABELS)
        predictions={phase:sum((results[phase][items[0]['video_name']]['scores'][key]-results[phase][items[1]['video_name']]['scores'][key])*weights[key] for key in LABELS) for phase in results}
        per_group.append({'videos':[x['video_name'] for x in items],'prompt_sha256':hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                          'mae':errors,'detail_minus_baseline_mae':errors['more_detail']-errors['baseline'],'annotation_margin':truth_margin,
                          'predicted_margin':predictions,'direction_correct':{phase:None if abs(truth_margin)<1e-8 else truth_margin*margin>1e-8 for phase,margin in predictions.items()}})
    for item in data['samples']:
        name=item['video_name'];pred=results['more_detail'][name]['scores']
        rows.append({'video_name':name,'ground_truth':item['ground_truth'],'prediction':pred,
                     'absolute_errors':{key:abs(pred[key]-float(item['ground_truth'][key])) for key in LABELS}})
    changes=[row['detail_minus_baseline_mae'] for row in per_group]
    ranks=[row['direction_correct'] for row in per_group if row['direction_correct']['baseline'] is not None]
    overview={'prompt_groups_improved':sum(x<-1e-8 for x in changes),'prompt_groups_unchanged':sum(abs(x)<=1e-8 for x in changes),'prompt_groups_regressed':sum(x>1e-8 for x in changes),
              'ranking_fixed':sum(not x['baseline'] and x['more_detail'] for x in ranks),'ranking_regressed':sum(x['baseline'] and not x['more_detail'] for x in ranks),
              'ranking_wrong_in_both':sum(not x['baseline'] and not x['more_detail'] for x in ranks),'ranking_right_in_both':sum(x['baseline'] and x['more_detail'] for x in ranks)}
    output={'overview':overview,'prompt_groups':per_group,'score_histograms':histograms,'more_detail_samples':rows,
            'source_summary_sha256':hashlib.sha256((report/'summary.json').read_bytes()).hexdigest()}
    gains=[-row['detail_minus_baseline_mae'] for row in per_group];net=sum(gains);best=max(gains)
    output.update(ground_truth_histograms={key:dict(collections.Counter(item['ground_truth'][key] for item in data['samples'])) for key in LABELS},largest_group_share_of_net_mae_gain=best/net if abs(net)>1e-8 else None,mean_mae_difference_without_best_group=-(net-best)/(len(per_group)-1) if len(per_group)>1 else None)
    (report/'distribution-analysis.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print('DISTRIBUTION',json.dumps(overview));print('GROUP_CHANGES',[(row['videos'],round(row['detail_minus_baseline_mae'],3),row['direction_correct']) for row in per_group]);print('HISTOGRAMS',json.dumps(histograms))
    session=read(report/'session.json');print('RUNTIME_MINUTES',sum(a['wall_seconds'] for a in session['attempts'])/60)
if __name__=='__main__': main()
