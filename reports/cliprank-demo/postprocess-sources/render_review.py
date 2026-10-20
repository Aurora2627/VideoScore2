"""Render a standalone Chinese review dashboard from saved candidate decisions."""
import argparse,html,json
from pathlib import Path

def render(run_dir,output):
    if output.exists():raise FileExistsError(output)
    selection=json.loads((run_dir/'selection.json').read_text(encoding='utf-8'))
    flat=[row for group in selection['groups'] for row in group['candidates']]
    escape=lambda x:html.escape(str(x),quote=True)
    names={'pending':'等待评估','comparison_pending':'等待同组评估','manual_review':'待人工复核','single_candidate_review':'单个素材待确认','shortlist':'候选短名单','reserve':'备选'}
    labels=('visual quality','text-to-video alignment','physical/common-sense consistency')
    labels_cn=('画面质量','文本匹配','物理合理性')
    head='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ClipRank · 视频素材评审</title><style>
    :root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#18253c;background:#f3f6fa}*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:36px 24px}header{background:#122542;color:white;padding:30px;border-radius:20px}h1{font-size:28px;margin:10px 0}header p{color:#c4d4e9;line-height:1.7}.eyebrow{letter-spacing:3px;color:#89c6ff;font-size:12px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:22px 0}.stat,section{background:white;border:1px solid #e0e7f0;border-radius:14px;padding:22px}.stat strong{font-size:30px;display:block}.stat span,.muted{color:#64748b;font-size:14px}nav{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}input,select{border:1px solid #ccd6e5;background:white;border-radius:9px;padding:12px;font:inherit}input{flex:1;min-width:200px}section{margin:18px 0}h2{font-size:18px;margin:0 0 12px}.prompt{line-height:1.7;color:#475569;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:14px 8px;border-bottom:1px solid #edf1f6;vertical-align:top}th{color:#64748b;font-weight:500}.badge{display:inline-block;border-radius:6px;padding:5px 9px;background:#edf2f8;white-space:nowrap}.manual_review{background:#fff0df;color:#99520d}.shortlist{background:#dff5eb;color:#126e46}.reasons{font-size:12px;color:#8c591d;margin-top:7px}.analysis{margin:14px 0;font-size:14px}.analysis summary{cursor:pointer;color:#285da0}.analysis pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.7;max-height:420px;overflow:auto;background:#f7f9fc;padding:16px;border-radius:10px}footer{color:#64748b;font-size:13px;line-height:1.8;margin:28px 0}[hidden]{display:none!important}@media(max-width:700px){.stats{grid-template-columns:repeat(2,1fr)}.table{overflow-x:auto}main{padding:16px}header{padding:22px}}
    </style><main><header><div class="eyebrow">CLIPRANK / CREATIVE REVIEW</div><h1>AIGC 短视频素材评审台</h1><p>围绕同一创意需求比较候选视频，集中检查画面、内容与动作缺陷。<br>系统提供辅助评估；最终采用与发布由创作者确认。</p></header>'''
    cards=[('视频素材',len(flat)),('创意需求',len(selection['groups'])),('候选短名单',sum(r['selected'] for r in flat)),('待人工复核',sum(r['decision']=='manual_review' for r in flat))]
    body=[head,'<div class="stats">']
    body+=['<div class="stat"><strong>'+str(count)+'</strong><span>'+name+'</span></div>' for name,count in cards]
    body+=['</div><nav><input id="search" placeholder="搜索提示词或视频名称" aria-label="搜索提示词或视频名称"><select id="filter" aria-label="筛选处理建议"><option value="all">全部素材</option><option value="pending">等待评估</option><option value="comparison_pending">等待同组评估</option><option value="shortlist">候选短名单</option><option value="manual_review">待人工复核</option><option value="single_candidate_review">单个素材待确认</option><option value="reserve">备选</option></select></nav>']
    for index,group in enumerate(selection['groups'],1):
        body+=['<section data-group><h2>创意需求 '+str(index)+'</h2><p class="prompt">'+escape(group['prompt'])+'</p><p class="muted">'+('在同一提示词的有效候选中进行比较。' if group['comparison_available'] else '有效候选不足两个，仅评估，暂不排名。')+'</p><div class="table"><table><thead><tr><th>视频</th><th>画面 / 文本 / 物理</th><th>规则总分</th><th>组内排名</th><th>处理建议</th></tr></thead><tbody>']
        for row in group['candidates']:
            scores=row.get('scores') or {};score_text=' / '.join(str(scores.get(k,'—')) for k in labels)
            reasons=[]
            for reason in row.get('review_reasons',[]):
                for old,new in zip(labels,labels_cn):reason=reason.replace(old+' below minimum',new+'低于初筛门槛')
                reasons.append(reason.replace('Incomplete or invalid model output','模型输出不完整或评分无效'))
            weighted='—' if row['weighted_score'] is None else format(row['weighted_score'],'.2f')+' / 5'
            body+=['<tr data-item data-decision="'+escape(row['decision'])+'" data-search="'+escape(group['prompt']+' '+row['video_name'])+'"><td><strong>'+escape(row['video_name'])+'</strong></td><td>'+escape(score_text)+'</td><td>'+weighted+'</td><td>'+('未排名' if row['rank'] is None else str(row['rank']))+'</td><td><span class="badge '+escape(row['decision'])+'">'+names[row['decision']]+'</span><div class="reasons">'+escape('；'.join(reasons))+'</div></td></tr>']
        body+=['</tbody></table></div>']
        for row in group['candidates']:
            media=run_dir/'media'/(row['video_name']+'.mp4')
            if media.exists():body+=['<details class="analysis" data-item data-decision="'+escape(row['decision'])+'" data-search="'+escape(group['prompt']+' '+row['video_name'])+'"><summary>'+escape(row['video_name'])+' · 查看视频</summary><video controls preload="none" style="max-width:100%;max-height:360px;margin-top:14px" src="media/'+escape(row['video_name'])+'.mp4"></video></details>']
            raw=run_dir/row['video_name']/'raw_output.txt';text=raw.read_text(encoding='utf-8') if raw.exists() else '未生成完整分析，请检查该素材的运行记录。'
            body+=['<details class="analysis" data-item data-decision="'+escape(row['decision'])+'" data-search="'+escape(group['prompt']+' '+row['video_name'])+'"><summary>'+escape(row['video_name'])+' · 查看模型分析</summary><pre>'+escape(text)+'</pre></details>']
        body+=['</section>']
    minimum='、'.join(n+' '+str(selection['policy']['minimum_scores'][k])+' 分' for n,k in zip(labels_cn,labels))
    weights='、'.join(n+' '+format(selection['policy']['weights'][k]*100,'.0f')+'%' for n,k in zip(labels_cn,labels))
    body+=['<footer>当前初筛门槛：'+escape(minimum)+'；总分权重：'+escape(weights)+'。权重与阈值尚未经过业务校准。<br>同分候选共同保留，不能将分数解读为采用概率。当前报告为小样本诊断，不构成线上效果证明。</footer><script>const search=document.getElementById("search"),filter=document.getElementById("filter");function apply(){const q=search.value.toLowerCase();document.querySelectorAll("[data-group]").forEach(group=>{let visible=0;group.querySelectorAll("[data-item]").forEach(row=>{const show=row.dataset.search.toLowerCase().includes(q)&&(filter.value==="all"||row.dataset.decision===filter.value);row.hidden=!show;if(show)visible++});group.hidden=!visible})}search.addEventListener("input",apply);filter.addEventListener("change",apply);</script></main></html>']
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(chr(10).join(body),encoding='utf-8')
    return {'video_count':len(flat),'group_count':len(selection['groups']),'shortlist':sum(r['selected'] for r in flat)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(render(a.run_dir,a.output)))
if __name__=='__main__':main()
