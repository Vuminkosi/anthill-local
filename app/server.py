import os, json, threading, uuid, traceback
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from pipeline import run_job

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100000
jobs = {}
lock = threading.Lock()
render_lock = threading.Lock()

HTML = '''<!doctype html><html><head><meta name="viewport" content="width=device-width, initial-scale=1"><title>AntHill Video Factory</title><style>
body{font-family:system-ui;background:#141820;color:#f3f4f6;max-width:780px;margin:40px auto;padding:0 18px}h1{font-size:32px}p{color:#bdc4d0}textarea{width:100%;height:210px;background:#252c38;color:white;border:1px solid #506079;border-radius:12px;padding:14px;box-sizing:border-box;font-size:16px}button{margin-top:15px;background:#ffd166;color:#111;padding:14px 24px;border:0;border-radius:9px;font-weight:bold;cursor:pointer}select{padding:8px}.card{background:#232b37;border-radius:14px;padding:20px;margin-top:20px}a{color:#ffd166}code{color:#9be8d6}</style></head><body><h1>🐜 AntHill Video Factory</h1><p>Local production • 30-second 9:16 animated story • AI images, moving camera, narrator, captions and original ambient music.</p><label>Story to dramatize</label><textarea id="story" placeholder="Paste a story here. Avoid private identifying information.">A man recovering from surgery felt terrified about a possible cancer diagnosis. When he went to his usual breakfast diner, a kind waitress quietly paid for his meal. He cried at the unexpected kindness. Two days later, his doctor told him he was cancer-free. The man never forgot her kindness.</textarea><div><button id="generate">Generate 30-second video</button></div><div id="result" class="card" style="display:none"></div><p>First launch downloads local AI and voice models. Internet is not needed for subsequent jobs once models are cached. Processing may take a long time on a 6–8 GB GPU.</p><script>
const result=document.getElementById('result');const btn=document.getElementById('generate');
btn.onclick=async()=>{btn.disabled=true;result.style.display='block';result.textContent='Submitting job...';try{let r=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({story:document.getElementById('story').value})});let j=await r.json();if(!r.ok)throw Error(j.error);poll(j.id)}catch(e){result.textContent=e.message;btn.disabled=false}};
async function poll(id){let r=await fetch('/api/jobs/'+id), j=await r.json();result.innerHTML='<b>Status:</b> '+j.status+'<p>'+String(j.progress||'')+'</p>'+(j.status==='done'?'<a href="'+j.video+'">Download MP4</a>':'')+(j.status==='error'?'<pre>'+String(j.error).replaceAll('<','&lt;')+'</pre>':'');if(j.status==='done'||j.status==='error'){btn.disabled=false;return;}setTimeout(()=>poll(id),3000)}
</script></body></html>'''

@app.get('/')
def home():return render_template_string(HTML)

@app.post('/api/jobs')
def start():
    story=str((request.get_json(silent=True) or {}).get('story','')).strip()
    if len(story)<25:return jsonify(error='Please enter at least 25 characters.'),400
    job_id=uuid.uuid4().hex[:12]
    with lock:jobs[job_id]=dict(id=job_id,status='queued',progress='Queued')
    def work():
        def progress(message):
            with lock:jobs[job_id]['progress']=message
        try:
            with lock:jobs[job_id]['status']='working'
            with render_lock:
                path=run_job(job_id,story,progress)
            with lock:jobs[job_id].update(status='done',video='/videos/'+os.path.basename(path),progress='Finished')
        except Exception as exc:
            with lock:jobs[job_id].update(status='error',error=str(exc),progress='Failed; check container logs for details')
            traceback.print_exc()
    threading.Thread(target=work,daemon=True).start()
    return jsonify(id=job_id,status='queued'),202

@app.get('/api/jobs/<id>')
def status(id):
    with lock:job=jobs.get(id)
    return jsonify(job) if job else (jsonify(error='Not found'),404)

@app.get('/videos/<path:name>')
def video(name):return send_from_directory('/data/output',name,as_attachment=True)

if __name__=='__main__':app.run(host='0.0.0.0',port=8080,threaded=True)
