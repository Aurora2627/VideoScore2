"""Build a local visual audit of unresolved ranking pairs; never sends labels to a model."""
import collections,hashlib,html,json,shutil
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1]
LABELS=('visual quality','text-to-video alignment','physical/common-sense consistency')
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def contact(video,indices,destination):
    capture=cv2.VideoCapture(str(video));cells=[]
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read()
            if not ok:raise ValueError('Cannot decode frame '+str(index))
            scale=min(300/frame.shape[1],180/frame.shape[0]);image=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
            cell=np.full((208,304,3),245,dtype=np.uint8);top=4+(180-image.shape[0])//2;left=(304-image.shape[1])//2
            cell[top:top+image.shape[0],left:left+image.shape[1]]=image
            cv2.putText(cell,'frame '+str(index),(8,201),cv2.FONT_HERSHEY_SIMPLEX,.5,(25,25,25),1,cv2.LINE_AA);cells.append(cell)
    finally:capture.release()
    while len(cells)%4:cells.append(np.full_like(cells[0],245))
    sheet=np.vstack([np.hstack(cells[i:i+4]) for i in range(0,len(cells),4)])
    assert cv2.imwrite(str(destination),sheet)
def main():
    source=ROOT/'reports/expanded-evaluation';audit=read(source/'distribution-analysis.json')
    failed=[group for group in audit['prompt_groups'] if group['direction_correct']['more_detail'] is False]
    manifest=read(source/'manifest.json');items={x['video_name']:x for x in manifest['samples']}
    metadata_root=Path('D:/career-projects/data/VideoScoreBench-v2');provenance=read(metadata_root/'metadata-provenance.json')
    assert sha(metadata_root/'metadata.json')==provenance['sha256']
    metadata={x['video_name']:x for x in read(metadata_root/'metadata.json')}
    out=ROOT/'reports/error-review';out.mkdir(parents=True,exist_ok=True);media=out/'media'
    if media.exists():raise FileExistsError('Review media already exists; retain it or choose a new checkout to rebuild without overwriting evidence')
    media.mkdir()
    intro='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ClipRank 排序错误复核</title><style>body{font:16px system-ui;margin:24px;max-width:1500px;color:#182438}article{border:1px solid #ccd5e3;padding:18px;margin:20px 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%;height:auto}video{width:100%;max-height:280px}pre{white-space:pre-wrap;font:14px system-ui}h1,h2,h3{margin:12px 0}a{color:#235bb9}</style>'
    documents=[];records=[]
    for index,group in enumerate(failed,1):
        parts=[intro,'<h1>候选对 '+str(index)+'</h1><p>'+html.escape(items[group['videos'][0]]['prompt'])+'</p><p>标注方向差：'+str(round(group['annotation_margin'],3))+'；448 预测方向差：'+str(round(group['predicted_margin']['more_detail'],3))+'</p><div class="pair">']
        for name in group['videos']:
            item=items[name];result=read(source/'more_detail'/name/'result.json');sampling=result['video_sampling'];video=Path(item['video_path']);assert sha(video)==item['video_sha256']
            sparse=sampling['frame_indices'];dense=np.linspace(0,sampling['total_frames']-1,min(12,sampling['total_frames']),dtype=int).tolist()
            contact(video,sparse,media/(name+'_sampled.jpg'));contact(video,dense,media/(name+'_timeline.jpg'));shutil.copy2(video,media/(name+'.mp4'))
            raw=(source/'more_detail'/name/result['artifact_directory']/'raw_output.txt').read_text(encoding='utf-8')
            thinking=metadata[name]['thinking'];thinking=thinking if isinstance(thinking,str) else json.dumps(thinking,ensure_ascii=False)
            record={'video_name':name,'prompt':item['prompt'],'ground_truth':item['ground_truth'],'predicted_scores':result['scores'],'sampling':sampling,
                    'video_sha256':item['video_sha256'],'dense_review_indices':dense,'raw_output':raw,'dataset_reasoning':thinking}
            records.append(record)
            scores=lambda x:' / '.join(str(x[k]) for k in LABELS)
            parts+=['<article><h2>'+name+'</h2><p>画质 / 文本 / 物理：标注 '+scores(item['ground_truth'])+'；预测 '+scores(result['scores'])+'</p>',
              '<video controls preload="metadata" src="media/'+name+'.mp4"></video><h3>实际送入模型的 4 个采样位置</h3><img src="media/'+name+'_sampled.jpg"><h3>较密时间线复核（不是模型输入）</h3><img src="media/'+name+'_timeline.jpg">',
              '<h3>模型原始解释</h3><pre>'+html.escape(raw)+'</pre><details><summary>数据集解释（仅供复核，非模型输入）</summary><pre>'+html.escape(thinking)+'</pre></details></article>']
        parts+=['</div></html>'];(out/('pair_'+str(index)+'.html')).write_text(chr(10).join(parts),encoding='utf-8');documents.append('<li><a href="pair_'+str(index)+'.html">候选对 '+str(index)+'：'+' / '.join(group['videos'])+'</a></li>')
    (out/'review.html').write_text(intro+'<h1>ClipRank 排序错误复核</h1><p>4 组候选，8 段视频。较密时间线用于观察遗漏；不能仅凭静态截图确认运动或物理是否正确。标注解释与模型解释均须核验。</p><ul>'+''.join(documents)+'</ul></html>',encoding='utf-8')
    (media/'review-records.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'review-index.json').write_text(json.dumps({'source_summary_sha256':sha(source/'summary.json'),'groups':failed,'review_record_sha256':sha(media/'review-records.json'),'media_in_git':False},indent=2),encoding='utf-8')
    print('ERROR_REVIEW_READY',len(failed),len(records),str(out),flush=True)
if __name__=='__main__':main()
