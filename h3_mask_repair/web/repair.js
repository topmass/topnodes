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
   this.addWidget('button', 'Open mask repair', null, () => openEditor(this, widget), { serialize: false });
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
 dialog.innerHTML = `<h2></h2><p>Run the mask stage first. Green adds the target; red excludes another area. Paint and Erase work without SAM.</p>
 <canvas style="width:100%;height:auto;touch-action:none;cursor:crosshair"></canvas>
 <div style="display:flex;gap:8px;align-items:center;margin:12px 0"><button data-act="prev">Previous</button><input aria-label="Video frame" type="range" min="0" value="0" style="flex:1"><button data-act="next">Next</button><output></output></div>
 <div style="display:flex;gap:10px;flex-wrap:wrap"><label>Tool <select aria-label="Repair tool"><option value="positive">Add target point</option><option value="negative">Exclude point</option><option value="paint">Paint mask</option><option value="erase">Erase mask</option></select></label>
 <label>Brush <input aria-label="Brush size" type="number" value="20" min="1" max="200" style="width:60px"> px</label>
 <label>Repair range <select aria-label="Repair range"><option value="forward">From this frame to the end</option><option value="single">This frame only</option><option value="custom">Through a chosen frame</option></select></label>
 <label>Through frame <input aria-label="Last repair frame" type="number" value="1" min="1" style="width:80px"></label>
 <button data-act="apply">Apply and preview mask only</button><button data-act="draft">Clear new marks</button><button data-act="undo">Undo last repair</button><button data-act="reset-frame">Reset this frame to SAM</button><button data-act="clear">Clear all repairs</button><button data-act="close">Keep edits and close</button></div>
 <p role="status"></p><small>Frame numbers start at 1. Choose a repair range, then click Apply. You can correct a later failure without changing earlier frames. Reset this frame restores its original selected SAM mask. Tracking replaces masks only in the selected range. Save the workflow to keep edits. Original SAM saves stay unchanged.</small>`;
 dialog.querySelector('h2').textContent = node.title;
 document.body.append(dialog); dialog.showModal();
 const canvas=dialog.querySelector('canvas'), ctx=canvas.getContext('2d'), slider=dialog.querySelector('[type=range]');
 const range=dialog.querySelector('[aria-label="Repair range"]');
 const [brush,end]=dialog.querySelectorAll('[type=number]'), status=dialog.querySelector('[role=status]');
 let draft={positive:[],negative:[],strokes:[]}, frame=0, base=null, overlay=null, drawing=null, loadId=0;
 const spec=()=>JSON.parse(widget.value || '{}');
 const save=s=>{widget.value=JSON.stringify(s); widget.callback?.(widget.value); app.graph.setDirtyCanvas(true);};
 const blank=()=>{draft={positive:[],negative:[],strokes:[]};drawing=null;};
 function draw() {
  if (!base) return;
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(base,0,0);
  ctx.globalAlpha=.45;ctx.drawImage(overlay,0,0);ctx.globalAlpha=1;
  for (const s of draft.strokes) {
   ctx.strokeStyle=s.mode==='paint'?'#4fffb0':'#ff495b';ctx.lineWidth=s.width*canvas.width;ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();
   s.points.forEach((p,i)=>i?ctx.lineTo(p.x*canvas.width,p.y*canvas.height):ctx.moveTo(p.x*canvas.width,p.y*canvas.height));
   if(s.points.length===1){const p=s.points[0];ctx.lineTo(p.x*canvas.width+.1,p.y*canvas.height);}ctx.stroke();
  }
  for(const key of ['positive','negative'])for(const p of draft[key]){ctx.fillStyle=key==='positive'?'#00ff80':'#ff3344';ctx.beginPath();ctx.arc(p.x*canvas.width,p.y*canvas.height,5,0,Math.PI*2);ctx.fill();}
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
  const token=++loadId;frame=Math.min(frame,info.frames-1);slider.max=info.frames-1;slider.value=frame;end.max=info.frames;updateRange();
  dialog.querySelector('output').textContent=`${frame+1} / ${info.frames} (${(frame/24).toFixed(2)}s)`;
  const load=name=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(Error('Preview expired. Run the mask stage again.'));im.src=api.apiURL('/view?'+new URLSearchParams({type:'temp',subfolder:info.directory,filename:`${name}-${frame}.${name==='frame'?'jpg':'png'}`}));});
  try{
   const [im,mask]=await Promise.all([load('frame'),load('mask')]);if(token!==loadId)return;
   canvas.width=info.width;canvas.height=info.height;canvas.style.width=`min(100%, ${48*info.width/info.height}vh)`;base=im;overlay=document.createElement('canvas');overlay.width=info.width;overlay.height=info.height;
   const c=overlay.getContext('2d');c.drawImage(mask,0,0);const pixels=c.getImageData(0,0,info.width,info.height);
   for(let i=0;i<pixels.data.length;i+=4){pixels.data[i+3]=pixels.data[i];pixels.data[i]=40;pixels.data[i+1]=240;pixels.data[i+2]=160;}c.putImageData(pixels,0,0);draw();status.textContent='Ready. Corrections are saved in this workflow.';
  }catch(e){status.textContent=e.message;}
 }
 node.refreshRepair=refresh;
 function change(value){frame=Math.max(0,Math.min(Number(slider.max),value));blank();refresh();}
 slider.oninput=()=>change(Number(slider.value));
 function point(e){const r=canvas.getBoundingClientRect();return{x:Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),y:Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))};}
 canvas.onpointerdown=e=>{
  if(!base)return;canvas.setPointerCapture(e.pointerId);const tool=dialog.querySelector('select').value,p=point(e);
  if(tool==='positive'||tool==='negative')draft[tool].push(p);
  else {drawing={mode:tool,width:Math.min(200,Math.max(1,Number(brush.value)))/canvas.width,points:[p]};draft.strokes.push(drawing);}draw();
 };
 canvas.onpointermove=e=>{if(drawing){drawing.points.push(point(e));draw();}};
 canvas.onpointerup=canvas.onpointercancel=()=>{drawing=null;};
 async function queueRepair(){
  const p=await app.graphToPrompt(),id=String(node.id),output={},visit=k=>{if(output[k])return;if(!p.output[k])throw Error('Repair node is disabled. Enable it first.');output[k]=p.output[k];for(const v of Object.values(output[k].inputs))if(Array.isArray(v)&&typeof v[1]==='number'&&p.output[String(v[0])])visit(String(v[0]));};visit(id);
  output['h3_repair_mask_image']={class_type:'MaskToImage',inputs:{mask:[id,0]}};
  output['h3_repair_preview']={class_type:'PreviewImage',inputs:{images:['h3_repair_mask_image',0]}};
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
    if(!Number.isInteger(last)||last<frame||last>=node.repairPreview.frames)throw Error('Choose an end frame at or after this frame.');
    if(draft.negative.length&&!draft.positive.length)throw Error('Add a green target point too, or use Erase.');
    if(s.steps?.length&&s.signature!==node.repairPreview.signature)throw Error('Source changed. Clear all repairs first.');
    save({signature:node.repairPreview.signature,steps:[...(s.steps||[]),{frame,end:last,...draft}]});blank();await queueRepair();
   }
   if(action==='reset-frame'){if(!node.repairPreview)throw Error('Run the mask stage first.');const s=spec();if(s.steps?.length&&s.signature!==node.repairPreview.signature)throw Error('Source changed. Clear all repairs first.');save({signature:node.repairPreview.signature,steps:[...(s.steps||[]),{frame,end:frame,restore:true}]});blank();await queueRepair();}
   if(action==='undo'){const s=spec();s.steps?.pop();save(s);blank();await queueRepair();}
   if(action==='clear'){save({});blank();await queueRepair();}
   if(action==='close')dialog.close();
  }catch(err){status.textContent=err.message;}
 };
 dialog.onclose=()=>{delete node.refreshRepair;dialog.remove();};refresh();
}
