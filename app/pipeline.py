import os, json, re, subprocess, wave, math, struct, time
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import numpy as np

ROOT=Path('/data'); OUT=ROOT/'output'; MODELS=ROOT/'models'; ASSETS=ROOT/'assets'
VOICE_ID='en_US-lessac-medium'
VOICE_BASE='https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

def command(args, stdin=None):
    p=subprocess.run(args,input=stdin,capture_output=True,text=isinstance(stdin,str),check=False)
    if p.returncode:raise RuntimeError('Command failed: '+str(args[:4])+' '+str(p.stderr)[-1200:])
    return p

def storyboard(story):
    prompt=("Return ONLY valid JSON containing keys narration (60-72 English words, 30-second emotional story, no invented claims) "
            "and scenes (exactly 6 objects, each with image and caption strings). The image descriptions must show the SAME recurring "
            "characters with detailed consistent clothing, location and appearance, 3D animated cartoon film still, cinematic lighting, 9:16 vertical. "
            "captions should be 3-9 words each. Follow story faithfully, don't add unfounded facts. Story:\n"+story[:6000])
    url=os.getenv('OLLAMA_URL','http://ollama:11434')+'/api/generate'
    for attempt in range(8):
        try:
            r=requests.post(url,json={'model':os.getenv('OLLAMA_MODEL','llama3.2:3b'),'prompt':prompt,'stream':False,'format':'json','options':{'temperature':0.4,'num_predict':1200}},timeout=360)
            r.raise_for_status(); obj=json.loads(r.json()['response']); scenes=obj['scenes']
            if len(scenes)!=6:raise ValueError('Model produced wrong number of scenes')
            if not obj.get('narration'):raise ValueError('Missing narration')
            return obj
        except Exception as e:
            if attempt==7:raise RuntimeError('Ollama storyboard failed. Run: docker compose exec ollama ollama pull llama3.2:3b. Last error: '+str(e))
            time.sleep(4)

def get_voice():
    d=MODELS/'voices';d.mkdir(parents=True,exist_ok=True)
    for ext in ('.onnx','.onnx.json'):
        f=d/(VOICE_ID+ext)
        if not f.exists():
            r=requests.get(VOICE_BASE+VOICE_ID+ext,timeout=180);r.raise_for_status()
            f.write_bytes(r.content)
    return d/(VOICE_ID+'.onnx')

def narrate(text, path):
    voice=get_voice()
    command(['python','-m','piper','--model',str(voice),'--output_file',str(path)],stdin=text)

def make_images(scenes,wd,progress):
    import torch
    from diffusers import AutoPipelineForText2Image
    model=os.getenv('IMAGE_MODEL','stabilityai/sd-turbo')
    progress('Loading image model (first run downloads several GB)...')
    pipe=AutoPipelineForText2Image.from_pretrained(model,torch_dtype=torch.float16,variant='fp16',safety_checker=None)
    # CPU offload is slower but appropriate for 6 GB VRAM; Ollama runs CPU-only.
    pipe.enable_model_cpu_offload()
    imgs=[]
    for i,s in enumerate(scenes):
        progress(f'Generating AI cartoon image {i+1}/6...')
        description=str(s.get('image',''))
        prompt=('stylized 3D cartoon film still, expressive emotion, warm cinematic lighting, detailed consistent characters, '
                'clean compositional depth, no writing, no captions. '+description)[:1000]
        image=pipe(prompt=prompt,height=896,width=512,num_inference_steps=2,guidance_scale=0.0).images[0]
        path=wd/f'image_{i:02d}.png';image.save(path);imgs.append(path)
    del pipe
    torch.cuda.empty_cache()
    return imgs

def card(image,caption,path):
    canvas=image.convert('RGB').resize((720,1280),Image.Resampling.LANCZOS)
    draw=ImageDraw.Draw(canvas,'RGBA');font=ImageFont.truetype(FONT,43)
    words=str(caption).split();lines=[];current=''
    for word in words:
        candidate=(current+' '+word).strip()
        if draw.textbbox((0,0),candidate,font=font)[2]>640 and current:lines.append(current);current=word
        else:current=candidate
    if current:lines.append(current)
    lines=lines[:3];h=65*len(lines);y=1060-h//2
    draw.rounded_rectangle((24,y-20,696,y+h+15),radius=22,fill=(0,0,0,170))
    for k,line in enumerate(lines):
        box=draw.textbbox((0,0),line,font=font);w=box[2]
        draw.text(((720-w)/2,y+k*65),line,font=font,fill=(255,255,255,255),stroke_width=2,stroke_fill=(0,0,0,255))
    canvas.save(path)

def duration(path):
    p=command(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)])
    return float(p.stdout.strip())

def music(path,seconds=30,sr=24000):
    # Original generated ambient musical bed; no copyrighted recording.
    samples=np.arange(seconds*sr,dtype=np.float64)/sr
    base=np.zeros_like(samples)
    for j,notes in enumerate(((220,261.63,329.63),(196,246.94,293.66),(174.61,220,261.63),(196,246.94,329.63))):
        select=(samples>=j*7.5)&(samples<(j+1)*7.5)
        t=samples[select]-j*7.5
        env=np.minimum(1,t/1.5)*np.minimum(1,(7.5-t)/2)
        base[select]=sum(np.sin(2*np.pi*f*t) for f in notes)*env/len(notes)
    pcm=(base*1200).astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr);f.writeframes(pcm.tobytes())

def render(scenes,images,voice,wd,final,progress):
    clips=[]
    for i,(scene,img) in enumerate(zip(scenes,images)):
        progress(f'Animating scene {i+1}/6 with camera movement...')
        captioned=wd/f'caption_{i}.png';card(Image.open(img),scene.get('caption',''),captioned)
        part=wd/f'part_{i:02d}.mp4'
        # Encode each 5-second segment, gentle zoom; no native AI motion claimed.
        vf="zoompan=z='min(zoom+0.00045,1.09)':d=150:s=720x1280:fps=30,format=yuv420p"
        command(['ffmpeg','-hide_banner','-loglevel','error','-y','-loop','1','-i',str(captioned),'-vf',vf,'-frames:v','150','-r','30','-an','-c:v','libx264','-preset','veryfast','-crf','22',str(part)])
        clips.append(part)
    listing=wd/'segments.txt';listing.write_text(''.join("file '"+str(p)+"'\n" for p in clips))
    silent=wd/'silent.mp4'
    command(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(silent)])
    bed=wd/'ambient.wav';music(bed)
    d=duration(voice);tempo=max(1.0,d/29.0)
    # atempo accepts 0.5-2.0; chain for arbitrarily long narrations
    filters=[]
    while tempo>2:filters.append('atempo=2.0');tempo/=2
    filters.append(f'atempo={tempo:.5f}')
    progress('Mixing narration, original ambient music and exporting MP4...')
    command(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',str(voice),'-i',str(bed),'-filter_complex',f'[1:a]{",".join(filters)},apad,atrim=duration=30[voice];[2:a]volume=0.32[music];[voice][music]amix=inputs=2:duration=first:normalize=0[a]','-map','0:v','-map','[a]','-t','30','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-movflags','+faststart',str(final)])

def run_job(job_id,story,progress):
    OUT.mkdir(parents=True,exist_ok=True);wd=OUT/('working_'+job_id);wd.mkdir(exist_ok=True)
    progress('Writing 30-second storyboard with Ollama...')
    plan=storyboard(story)
    (wd/'storyboard.json').write_text(json.dumps(plan,indent=2))
    progress('Generating local English narration...')
    voice=wd/'narration.wav';narrate(plan['narration'],voice)
    images=make_images(plan['scenes'],wd,progress)
    final=OUT/('AntHill_'+job_id+'.mp4')
    render(plan['scenes'],images,voice,wd,final,progress)
    return final
