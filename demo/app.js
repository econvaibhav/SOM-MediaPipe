import {infer,featuresFromLandmarks,StableGesture} from './model.mjs';
const $=id=>document.getElementById(id),canvas=$('hand-canvas'),ctx=canvas.getContext('2d'),video=$('camera-video');
let data,weights,mode='replay',paused=false,index=0,last=0,elapsed=0,raf=0,detector=null,stream=null,cameraGeneration=0,lastVideo=-1,previousSample=-1;
const latch=new StableGesture(),moves=[['OPEN / SHIELD','A palm raises a small ward.','▶'],['CLOSE / STRIKE','A fist makes the goblin flinch.','▶'],['POINTER / SPARK','A pointing finger casts a spark.','▶'],['OK / HEAL','The OK sign restores a little magic.','▶'],['HI / WAVE','The extra course class waves hello.','▶']];
function announce(text){$('status-message').textContent=text;$('status-message').hidden=!text;}
function resetReaction(){latch.reset();$('ward').hidden=true;$('spell-effect').className='';$('wizard').className='wizard';$('goblin').className='goblin';}
function act(id){
 resetReaction();latch.id=id;latch.fired=true;
 $('action-label').textContent=moves[id][0];$('action-note').textContent=moves[id][1];$('action-icon').textContent=moves[id][2];
 const wizard=$('wizard'),goblin=$('goblin'),effect=$('spell-effect');void wizard.offsetWidth;
 if(id===0)$('ward').hidden=false;
 if(id===1){wizard.classList.add('cast');goblin.classList.add('bump');effect.className='strike';}
 if(id===2){wizard.classList.add('cast');effect.className='spark';}
 if(id===3){wizard.classList.add('cast');effect.className='heal';}
 if(id===4)wizard.classList.add('wave');
}
function drawGrid(){
 ctx.fillStyle='#cbd3a3';ctx.fillRect(0,0,canvas.width,canvas.height);
 ctx.strokeStyle='#26312620';ctx.lineWidth=2;
 for(let x=40;x<canvas.width;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();}
 for(let y=20;y<canvas.height;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();}
}
function drawPoints(points,fit=false){
 let xy=points;
 if(fit){const minX=Math.min(...points.map(p=>p[0])),maxX=Math.max(...points.map(p=>p[0])),minY=Math.min(...points.map(p=>p[1])),maxY=Math.max(...points.map(p=>p[1]));const s=Math.min(560/(maxX-minX||1),450/(maxY-minY||1));xy=points.map(p=>[(p[0]-(minX+maxX)/2)*s+canvas.width/2,(p[1]-(minY+maxY)/2)*s+canvas.height/2-15]);}
 ctx.lineWidth=5;ctx.lineCap='square';ctx.strokeStyle='#263126';
 for(const[a,b]of data.connections){ctx.beginPath();ctx.moveTo(...xy[a]);ctx.lineTo(...xy[b]);ctx.stroke();}
 xy.forEach((p,i)=>{ctx.fillStyle=i===0?'#263126':'#7c8b5d';ctx.beginPath();ctx.arc(...p,i===0?9:6,0,Math.PI*2);ctx.fill();});
}
function updatePrediction(values,now){
 const scores=infer(values,weights),best=scores.indexOf(Math.max(...scores)),ranked=[...scores].sort((a,b)=>b-a),clear=scores[best]>=.7&&ranked[0]-ranked[1]>=.15;
 $('predicted-label').textContent=data.labels[best];$('prediction-state').textContent=clear?'MATCH':'WAIT';$('prediction-state').className=clear?'':'uncertain';
 scores.forEach((score,i)=>{const row=$(`score-${i}`);row.classList.toggle('leading',i===best);row.querySelector('.score-fill').style.width=`${score*100}%`;row.querySelector('.score-percent').textContent=`${(score*100).toFixed(1)}%`;});
 const fired=latch.update(clear?best:null,now);if(fired!==null)act(fired);
 if(!clear){$('ward').hidden=true;$('action-label').textContent='WAITING FOR A CLEAR POSE';$('action-note').textContent='The action pauses while the model is unsure.';$('action-icon').textContent='▷';}
 if(clear){$('action-label').textContent=moves[best][0];$('action-note').textContent=moves[best][1];$('action-icon').textContent=moves[best][2];}return {label:data.labels[best],scores,clear};
}
function selectExample(id){
 stopCamera();mode='replay';paused=false;index=data.frames.findIndex(f=>f.label===data.labels[id]);elapsed=0;previousSample=-1;resetReaction();updateControls();renderReplay(performance.now());
}
function updateControls(){
 $('camera-button').classList.toggle('selected',mode==='camera');$('camera-button').setAttribute('aria-pressed',String(mode==='camera'));
 $('replay-button').classList.toggle('selected',mode==='replay');$('replay-button').setAttribute('aria-pressed',String(mode==='replay'));
 $('pause-button').disabled=mode==='camera';$('pause-button').textContent=paused?'▶':'Ⅱ';$('pause-button').setAttribute('aria-label',paused?'Play replay':'Pause replay');
 $('input-label').textContent=mode==='camera'?'CAMERA':'REPLAY';if(!$('camera-button').disabled)$('camera-button').textContent=mode==='camera'?'STOP CAMERA':'CAMERA';
 $('source-note').textContent=mode==='camera'?'Show one hand. Camera frames stay in this browser.':'Recorded landmark rows running through the saved course model.';
}
function renderReplay(now){
 const frame=data.frames[index];drawGrid();drawPoints(Array.from({length:21},(_,i)=>[frame.values[i*2],frame.values[i*2+1]]),true);
 const id=data.labels.indexOf(frame.label);$('sample-label').textContent=`${frame.label} · dataset row ${frame.row.toLocaleString()}`;
 [...$('sample-buttons').children].forEach((b,i)=>{b.classList.toggle('active',i===id);b.setAttribute('aria-pressed',String(i===id));});
 if(previousSample!==id){resetReaction();previousSample=id;}
 updatePrediction(frame.values,now);
}
function noHand(){if(video.readyState>=2){ctx.save();ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();}else drawGrid();$('sample-label').textContent='SHOW ONE HAND';$('predicted-label').textContent='—';$('prediction-state').textContent='NO HAND';$('prediction-state').className='uncertain';latch.update(null,performance.now());$('ward').hidden=true;$('action-label').textContent='SHOW ONE HAND';$('action-note').textContent='Try a palm, fist, or pointing finger.';for(let i=0;i<5;i++){$(`score-${i}`).querySelector('.score-fill').style.width='0%';$(`score-${i}`).querySelector('.score-percent').textContent='—';}}
function stopCamera(){cameraGeneration++;stream?.getTracks().forEach(t=>t.stop());stream=null;detector?.close();detector=null;video.srcObject=null;$('camera-button').disabled=false;$('camera-button').textContent='CAMERA';}
async function useCamera(){
 if(mode==='camera'){stopCamera();mode='replay';updateControls();return;}
 const generation=++cameraGeneration;$('camera-button').disabled=true;$('camera-button').textContent='LOADING...';announce('Loading the hand tracker. Your browser will ask for camera access.');
 let task=null,localStream=null;
 try{
  if(!navigator.mediaDevices?.getUserMedia)throw new Error('Use HTTPS or localhost for camera access. Recorded examples are still available.');
  const {HandLandmarker,FilesetResolver}=await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/vision_bundle.mjs');
  const files=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22-rc.20250304/wasm');
  task=await HandLandmarker.createFromOptions(files,{baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',delegate:'CPU'},runningMode:'VIDEO',numHands:1,minHandDetectionConfidence:.65,minHandPresenceConfidence:.65,minTrackingConfidence:.6});
  if(generation!==cameraGeneration){task.close();return;}
  localStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:960},height:{ideal:540}},audio:false});
  if(generation!==cameraGeneration){task.close();localStream.getTracks().forEach(t=>t.stop());return;}
  detector=task;stream=localStream;video.srcObject=stream;await video.play();if(generation!==cameraGeneration)return;mode='camera';lastVideo=-1;resetReaction();updateControls();announce('');
 }catch(error){localStream?.getTracks().forEach(t=>t.stop());task?.close();if(generation!==cameraGeneration)return;detector=null;stream=null;mode='replay';updateControls();announce(error.name==='NotAllowedError'?'Camera permission was denied. Allow it in your browser, then try again. Recorded examples still work.':error.name==='NotFoundError'?'No camera was found. You can explore the recorded examples.':error.message||'The camera could not start. Try the recorded examples.');}
 finally{if(generation===cameraGeneration){$('camera-button').disabled=false;$('camera-button').textContent=mode==='camera'?'STOP CAMERA':'CAMERA';}}
}
function tick(now){
 if(now-last>=85){const dt=Math.min(now-last,150);last=now;
  if(mode==='replay'){if(!paused){elapsed+=dt;if(elapsed>=110){index=(index+1)%data.frames.length;elapsed=0;}}renderReplay(now);}
  else if(video.readyState>=2&&video.currentTime!==lastVideo){
   lastVideo=video.currentTime;
   try{const points=detector.detectForVideo(video,now).landmarks[0];
    if(!points)noHand();else{ctx.save();ctx.translate(canvas.width,0);ctx.scale(-1,1);ctx.drawImage(video,0,0,canvas.width,canvas.height);ctx.restore();drawPoints(points.map(p=>[(1-p.x)*canvas.width,p.y*canvas.height]));$('sample-label').textContent='Live hand · mirrored';[...$('sample-buttons').children].forEach(b=>{b.classList.remove('active');b.setAttribute('aria-pressed','false');});updatePrediction(featuresFromLandmarks(points,video.videoWidth,video.videoHeight),now);}
   }catch{stopCamera();mode='replay';updateControls();announce('Hand tracking stopped. Try enabling the camera again.');}
  }
 }
 raf=requestAnimationFrame(tick);
}
try{
 [data,weights]=await Promise.all(['replay.json','model.json'].map(async path=>{const r=await fetch(new URL(path,import.meta.url));if(!r.ok)throw new Error(`Could not load ${path}`);return r.json();}));
 data.labels.forEach((label,i)=>{const b=document.createElement('button');b.textContent=label;b.addEventListener('click',()=>selectExample(i));$('sample-buttons').append(b);const row=document.createElement('div');row.className='score-row';row.id=`score-${i}`;row.innerHTML='<span class="score-name"></span><span class="score-track"><span class="score-fill"></span></span><span class="score-percent">—</span>';row.querySelector('.score-name').textContent=label;$('score-bars').append(row);});
 $('replay-button').addEventListener('click',()=>{stopCamera();mode='replay';resetReaction();updateControls();announce('');});
 $('camera-button').addEventListener('click',useCamera);
 $('pause-button').addEventListener('click',()=>{paused=!paused;updateControls();});
 updateControls();renderReplay(performance.now());raf=requestAnimationFrame(tick);
 // Optional read-only WebMCP surface; no camera access can be triggered by an agent.
 const context=document.modelContext;if(context?.registerTool){try{Promise.resolve(context.registerTool({name:'read_gesture_demo',description:'Read the current SOM MediaPipe gesture prediction and input mode.',inputSchema:{type:'object',properties:{},additionalProperties:false},annotations:{readOnlyHint:true},execute:()=>({mode,label:$('predicted-label').textContent,state:$('prediction-state').textContent})})).catch(()=>{});}catch{}}
}catch(error){$('loading-error').hidden=false;$('loading-error').textContent='The demo data could not load. Serve this folder over HTTP or HTTPS, then reload.';$('pause-button').disabled=true;$('camera-button').disabled=true;announce(error.message);}
window.addEventListener('pagehide',()=>{cancelAnimationFrame(raf);stopCamera();});
