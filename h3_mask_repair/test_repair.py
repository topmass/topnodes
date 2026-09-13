"""Run: PYTHONPATH=. .venv/bin/python custom_nodes/h3_mask_repair/test_repair.py"""
import importlib.util
import tempfile
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('repair',Path(__file__).with_name('__init__.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
images=torch.zeros(5,16,16,3);masks=torch.zeros(5,16,16)
def segment(frame,pos,neg):
 assert pos==[{'x':8,'y':8}]
 return torch.ones(1,16,16)
def track(frames,seed):return seed.expand(len(frames),-1,-1).clone()
step={'frame':1,'end':3,'positive':[{'x':.5,'y':.5}]}
r=m.apply_mask_repairs(images,masks,[step],segment,track)
assert r[1:4].all() and not r[0].any() and not r[4].any() and not masks.any()
paint={'frame':2,'end':2,'strokes':[{'mode':'paint','width':.2,'points':[{'x':.5,'y':.5}]}]}
r=m.apply_mask_repairs(images,masks,[paint],segment,track)
assert r[2,8,8]==1 and not r[:2].any() and not r[3:].any()
for bad in [{'frame':0,'end':5},{'frame':0,'end':0,'negative':[{'x':.5,'y':.5}]}]:
 try:m.apply_mask_repairs(images,masks,[bad],segment,track)
 except ValueError:pass
 else:raise AssertionError('Invalid repair accepted')
with tempfile.TemporaryDirectory() as d:
 m.folder_paths.get_temp_directory=lambda:d
 result=m.H3MaskRepair().repair(images,masks,visible_frames=4)
 assert result['ui']['h3_mask_repair'][0]['frames']==4
 assert len(list(Path(d).rglob('frame-*.jpg')))==4
 try:m.H3MaskRepair().repair(images,masks,repairs='{"signature":"wrong","steps":[{"frame":0,"end":0}]}')
 except ValueError:pass
 else:raise AssertionError('Stale repair accepted')
print('PASS: range isolation, point coordinates, paint, invalid ranges, stale masks, preview count')

r=m.apply_mask_repairs(images,masks,[step,{'frame':2,'end':2,'restore':True}],segment,track)
assert r[1].all() and not r[2].any() and r[3].all()
r=m.apply_mask_repairs(images,masks,[step,{'frame':3,'end':4,'positive':[{'x':.5,'y':.5}]}],segment,track)
assert not r[0].any() and r[1:].all()
print('PASS: reset one frame preserves neighbors; later forward repairs preserve earlier frames')

from custom_nodes.MaskVidExperiments.nodes_subject_crop import MVEx_SubjectCropNode, MVEx_SubjectUncropNode
source=torch.zeros(5,64,64,3)
partial=torch.zeros(5,64,64);partial[2,20:40,20:40]=1
mode={'mode':'auto','crop_scale':1.75,'padding':'firm','prefer':'stillness','aspect_ratio':0.0,'seamless_loop':False}
crop=MVEx_SubjectCropNode.execute(source,partial,mode,32,0).result
frame_gate=m.H3MaskFrameGate().gate(crop[1])[0]
output=MVEx_SubjectUncropNode.execute(torch.ones_like(crop[0]),source,crop[2],0,frame_gate).result[0]
assert output[2].any() and torch.equal(output[[0,1,3,4]],source[[0,1,3,4]])
assert not m.H3MaskFrameGate().gate(torch.zeros_like(partial))[0].any()
print('PASS: native crop/uncrop accepts isolated masked frame; empty frames remain pixel-identical')
