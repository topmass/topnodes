import { app } from '/scripts/app.js';
import { api } from '/scripts/api.js';

app.registerExtension({
 name: 'h3.mask.repair',
 async beforeRegisterNodeDef(type, data) {
  if (data.name !== 'H3MaskRepair') return;
  const created = type.prototype.onNodeCreated;
  type.prototype.onNodeCreated = function () {
   created?.apply(this, arguments);
   const widget = this.widgets.find(w => w.name === 'repairs');
   widget.type = 'hidden'; widget.computeSize = () => [0, -4];
   if (widget.inputEl) widget.inputEl.style.display = 'none';
   this.addWidget('button', 'Open mask suite', null, () => openEditor(this, widget), { serialize: false });
  };
  const executed = type.prototype.onExecuted;
  type.prototype.onExecuted = function (message) {
   executed?.apply(this, arguments);
   if (message.h3_mask_repair) {
    this.repairPreview = message.h3_mask_repair[0];
    this.refreshRepair?.();
   }
  };
 }
});

function openEditor(node, widget) {
 const dialog = document.createElement('dialog');
 dialog.style.cssText = 'width:min(1000px,95vw);background:#20242b;color:white;padding:20px;border:1px solid #667;border-radius:12px';
 dialog.innerHTML = `<h2></h2><p>Add: click a target or draw a mask. Erase: click a connected mask area to remove it, or drag to erase part of it.</p>
 <div><label>View <select aria-label="Mask view"><option value="overlay">Mask over source video</option><option value="mask">Mask only</option><option value="source">Source only</option></select></label> <label><input type="checkbox" aria-label="Highlight all mask pixels" checked> Highlight even faint mask pixels</label> <label>Zoom <select aria-label="Mask zoom"><option value="1">Fit</option><option value="2">2x</option><option value="4">4x</option></select></label> <strong data-pixels></strong></div>
 <div style="max-height:55vh;overflow:auto" data-viewport><canvas style="width:100%;height:auto;touch-action:none;cursor:crosshair"></canvas></div>
 <div style="display:flex;gap:8px;align-items:center;margin:12px 0"><button data-act="prev">Previous</button><input aria-label="Video frame" type="range" min="0" value="0" style="flex:1"><button data-act="next">Next</button><output></output></div>
 <div style="display:flex;gap:10px;flex-wrap:wrap"><label>Action <select aria-label="Repair tool"><option value="positive">Add mask</option><option value="erase">Erase mask</option></select></label>
 <label><input type="checkbox" aria-label="Replace entire frame mask"> Replace entire frame mask with my new selection/paint</label>
 <label>Brush <input aria-label="Brush size" type="number" value="20" min="1" max="200" style="width:60px"> px</label>
 <label>Repair range <select aria-label="Repair range"><option value="forward">From this frame to the end</option><option value="single">This frame only</option><option value="custom">Through a chosen frame</option></select></label>
 <label>Through frame <input aria-label="Last repair frame" type="number" value="1" min="1" style="width:80px"></label>
 <button data-act="apply">Apply</button><button data-act="draft">Clear new marks</button><button data-act="undo">Undo last change</button><button data-act="empty-frame">Empty this frame</button><button data-act="reset-frame">Reset this frame to SAM</button><button data-act="clear">Clear all repairs</button><button data-act="close">Done</button></div>
 <p role="status"></p><small>Frame numbers start at 1. Choose a repair range, then click Apply. You can correct a later failure without changing earlier frames. Reset this frame restores its original selected SAM mask. Tracking replaces masks only in the selected range. Save the workflow to keep edits. Original SAM saves stay unchanged.</small>`;
 dialog.querySelector('h2').textContent = node.title;
 document.body.append(dialog); dialog.showModal();
 const advanced=document.createElement('details');advanced.innerHTML='<summary>More options</summary>';dialog.append(advanced);
 for(const selector of ['[aria-label="Mask view"]','[aria-label="Highlight all mask pixels"]','[aria-label="Replace entire frame mask"]','[aria-label="Brush size"]','[aria-label="Last repair frame"]'])advanced.append(dialog.querySelector(selector).parentElement);
 for(const action of ['draft','empty-frame','reset-frame','clear'])advanced.append(dialog.querySelector(`[data-act="${action}"]`));
 advanced.append(dialog.querySelector('small'));
 const canvas=dialog.querySelector('canvas'), ctx=canvas.getContext('2d'), slider=dialog.querySelector('[type=range]');
 const view=dialog.querySelector('[aria-label="Mask view"]'), highlight=dialog.querySelector('[aria-label="Highlight all mask pixels"]'), zoom=dialog.querySelector('[aria-label="Mask zoom"]'), replace=dialog.querySelector('[aria-label="Replace entire frame mask"]');
 const range=dialog.querySelector('[aria-label="Repair range"]');
 const [brush,end]=dialog.querySelectorAll('[type=number]'), status=dialog.querySelector('[role=status]');
 let draft={positive:[],negative:[],strokes:[],erase_points:[],replace:false}, frame=0, base=null, overlay=null, drawing=null, loadId=0, maskImage=null;
 const spec=()=>JSON.parse(widget.value || '{}');
 const save=s=>{widget.value=JSON.stringify({...s,sessions:spec().sessions}); widget.callback?.(widget.value); app.graph.setDirtyCanvas(true);};
 const blank=()=>{draft={positive:[],negative:[],strokes:[],erase_points:[],replace:false};replace.checked=false;drawing=null;};
 const actionTool=dialog.querySelector('[aria-label="Repair tool"]');
 actionTool.onchange=()=>{blank();replace.disabled=actionTool.value==='erase';draw();};
 view.onchange=()=>draw();
 replace.onchange=()=>{draft.replace=replace.checked;draw();};
 zoom.onchange=()=>{canvas.style.width=`${Number(zoom.value)*Math.min(dialog.clientWidth-40,window.innerHeight*.48*canvas.width/canvas.height)}px`;};
 highlight.onchange=()=>{if(maskImage)makeOverlay();draw();};
 function makeOverlay(){const c=overlay.getContext('2d');c.clearRect(0,0,overlay.width,overlay.height);c.drawImage(maskImage,0,0);const pixels=c.getImageData(0,0,overlay.width,overlay.height);for(let i=0;i<pixels.data.length;i+=4){const value=pixels.data[i];pixels.data[i+3]=highlight.checked&&value>0?255:value;pixels.data[i]=40;pixels.data[i+1]=240;pixels.data[i+2]=160;}c.putImageData(pixels,0,0);}

 function draw() {
  if (!base) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='black';ctx.fillRect(0,0,canvas.width,canvas.height);if(view.value!=='mask')ctx.drawImage(base,0,0);
  if(view.value!=='source'&&!draft.replace){ctx.globalAlpha=view.value==='mask'?1:.6;ctx.drawImage(overlay,0,0);ctx.globalAlpha=1;}
  for (const s of draft.strokes) {
   ctx.strokeStyle=s.mode==='paint'?'#4fffb0':'#ff495b';ctx.lineWidth=s.width*canvas.width;ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();
   s.points.forEach((p,i)=>i?ctx.lineTo(p.x*canvas.width,p.y*canvas.height):ctx.moveTo(p.x*canvas.width,p.y*canvas.height));
   if(s.points.length===1){const p=s.points[0];ctx.lineTo(p.x*canvas.width+.1,p.y*canvas.height);}ctx.stroke();
  }
  for(const key of ['positive','negative','erase_points'])for(const p of draft[key]){ctx.fillStyle=key==='positive'?'#00ff80':'#ff3344';ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,5,0,Math.PI*2);ctx.fill();}
 }
 function updateRange() {
  end.disabled=range.value!=='custom';
  end.min=frame+1;
  end.value=range.value==='forward' ? node.repairPreview?.frames || frame+1 : range.value==='single' ? frame+1 : Math.max(frame+1,Number(end.value));
 }
 range.onchange=updateRange;
 async function refresh() {
  const info=node.repairPreview;
  if(!info){status.textContent='No preview yet. Run STEP 1, then open this editor.';return;}
  if(node.properties.editor_first && spec().signature!==info.signature){
   const s=spec(),sessions=s.sessions||{};
   if(s.signature)sessions[s.signature]=s.steps||[];
   widget.value=JSON.stringify({signature:info.signature,steps:sessions[info.signature]||[],sessions});
   save(spec());blank();
   if(spec().steps.length){await queueRepair();return;}
  }
  const token=++loadId;frame=Math.min(frame,info.frames-1);slider.max=info.frames-1;slider.value=frame;end.max=info.frames;updateRange();
  dialog.querySelector('[data-pixels]').textContent=info.mask_pixels ? `${info.mask_pixels[frame].toLocaleString()} masked source pixels${info.mask_pixels[frame]===0?' - EMPTY':''}` : '';
  base=null;
  dialog.querySelector('output').textContent=`${frame+1} / ${info.frames} (${(frame/24).toFixed(2)}s)`;
  const load=name=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(Error('Preview expired. Run the mask stage again.'));im.src=api.apiURL('/view?'+new URLSearchParams({type:'temp',subfolder:info.directory,filename:`${name}-${frame}.${name==='frame'?'jpg':'png'}`}));});
  try{
   const [im,mask]=await Promise.all([load('frame'),load('mask')]);if(token!==loadId)return;
   canvas.width=info.width;canvas.height=info.height;canvas.style.width=`min(100%, ${48*info.width/info.height}vh)`;base=im;overlay=document.createElement('canvas');overlay.width=info.width;overlay.height=info.height;
   maskImage=mask;makeOverlay();zoom.onchange();draw();status.textContent=info.repairs_skipped && !node.properties.editor_first ? 'New source or tracking result: old repairs were skipped. These are the new masks. Clear all repairs before adding new corrections; save a workflow copy first if you want to keep the old edits.' : 'Ready. Corrections are saved in this workflow.';
  }catch(e){status.textContent=e.message;}
 }
 node.refreshRepair=refresh;
 function change(value){frame=Math.max(0,Math.min(Number(slider.max),value));blank();refresh();}
 slider.oninput=()=>change(Number(slider.value));
 function point(e){const r=canvas.getBoundingClientRect();return{x:Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))};}
 canvas.onpointerdown=e=>{
  if(!base)return;canvas.setPointerCapture(e.pointerId);
  drawing={mode:actionTool.value==='erase'?'erase':'paint',width:Math.min(200,Math.max(1,Number(brush.value)))/canvas.width,points:[point(e)]};draft.strokes.push(drawing);draw();
 };
 canvas.onpointermove=e=>{if(drawing){const p=point(e),first=drawing.points[0];if(drawing.points.length>1||Math.hypot((p.x-first.x)*canvas.width,(p.y-first.y)*canvas.height)>3){drawing.points.push(p);draw();}}};
 canvas.onpointerup=()=>{if(drawing&&drawing.points.length===1){draft.strokes.pop();draft[actionTool.value==='erase'?'erase_points':'positive'].push(drawing.points[0]);}drawing=null;draw();};
 canvas.onpointercancel=()=>{drawing=null;};
 async function queueRepair(){
  const p=await app.graphToPrompt(),id=String(node.id),output={},visit=k=>{if(output[k])return;if(!p.output[k])throw Error('Repair node is disabled. Enable it first.');output[k]=p.output[k];for(const v of Object.values(output[k].inputs))if(Array.isArray(v)&&typeof v[1]==='number'&&p.output[String(v[0])])visit(String(v[0]));};visit(id);
  output['h3_repair_preview']={class_type:'ImageAndMaskPreview',inputs:{mask:[id,0],mask_opacity:.6,mask_color:'40,240,160',pass_through:true}};
  if(Object.values(output).some(n=>/Sampler|Guider/.test(n.class_type)))throw Error('Unexpected generation dependency. Repair queue stopped.');
  await api.queuePrompt(-1,{output,workflow:p.workflow});status.textContent='Repair queued. This runs mask processing only. The preview updates when complete.';
 }
 dialog.onclick=async e=>{
  const action=e.target.dataset.act;if(!action)return;
  try{
   if(action==='prev')change(frame-1);if(action==='next')change(frame+1);
   if(action==='draft'){blank();draw();}
   if(action==='apply'){
    if(!node.repairPreview)throw Error('Run the mask stage first.');
    const s=spec(),last=Number(end.value)-1;
    if(!draft.positive.length&&!draft.erase_points.length&&!draft.strokes.length&&!draft.replace)throw Error('Click or draw on the video first.');
    if(!Number.isInteger(last)||last<frame||last>=node.repairPreview.frames)throw Error('Choose an end frame at or after this frame.');
    if(draft.negative.length&&!draft.positive.length)throw Error('Add a green target point too, or use Erase.');
    if(s.steps?.length&&s.signature!==node.repairPreview.signature)throw Error('Source changed. Clear all repairs first.');
    save({signature:node.repairPreview.signature,steps:[...(s.steps||[]),{frame,end:last,...draft,erase_only:actionTool.value==='erase'}]});blank();await queueRepair();
   }
   if(action==='empty-frame'){if(!node.repairPreview)throw Error('Run the mask stage first.');const s=spec();if(s.steps?.length&&s.signature!==node.repairPreview.signature)throw Error('Source changed. Clear all repairs first.');save({signature:node.repairPreview.signature,steps:[...(s.steps||[]),{frame,end:frame,replace:true}]});blank();await queueRepair();}
   if(action==='reset-frame'){if(!node.repairPreview)throw Error('Run the mask stage first.');const s=spec();if(s.steps?.length&&s.signature!==node.repairPreview.signature)throw Error('Source changed. Clear all repairs first.');save({signature:node.repairPreview.signature,steps:[...(s.steps||[]),{frame,end:frame,restore:true}]});blank();await queueRepair();}
   if(action==='undo'){const s=spec();s.steps?.pop();save(s);blank();await queueRepair();}
   if(action==='clear'){save({});blank();await queueRepair();}
   if(action==='close')dialog.close();
  }catch(err){status.textContent=err.message;}
 };
 dialog.onclose=()=>{delete node.refreshRepair;dialog.remove();};
 if(node.properties.editor_first){status.textContent='Loading the clip for editing...';queueRepair().catch(err=>{status.textContent=err.message;});}
 else refresh();
}
