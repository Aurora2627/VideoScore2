"""Conservative supplementary gates; retain the original tied-pair outcome."""
def gates(original,supplement):
 dev=original['candidate_development']['more_detail'];reg=original['regression_candidate'];reg_base=original['regression_baseline'];old=original['fresh_candidate'];old_base=original['fresh_baseline'];new=supplement['phases']['more_detail']['metrics'];new_base=supplement['phases']['baseline']['metrics']
 complete=all(m['valid_rate']==1 for m in (dev,reg,old,old_base,new,new_base))
 mean=lambda a,b:(a['macro_mae']*a['sample_count']+b['macro_mae']*b['sample_count'])/(a['sample_count']+b['sample_count'])
 usable=complete and all(m['macro_mae'] is not None for m in (reg,reg_base,old,old_base,new,new_base))
 return {'all_outputs_complete':complete,'development_pair_correct':dev['pair_total']>0 and dev['pair_correct']==dev['pair_total'],'regression_not_worse':usable and reg['macro_mae']<=reg_base['macro_mae']+1e-8,'original_fresh_not_worse':usable and old['macro_mae']<=old_base['macro_mae']+1e-8,'supplement_fresh_not_worse':usable and new['macro_mae']<=new_base['macro_mae']+1e-8,'combined_fresh_strictly_better':usable and mean(old,new)<mean(old_base,new_base)-1e-8,'supplement_pair_correct':new['pair_total']>0 and new['pair_correct']==new['pair_total']}
