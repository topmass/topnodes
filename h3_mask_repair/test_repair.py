"""Run: PYTHONPATH=. .venv/bin/python custom_nodes/h3_mask_repair/test_repair.py"""
import json
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
 blank=m.H3MaskRepair().repair(images,visible_frames=4)
 assert torch.equal(blank['result'][0],masks)
 assert blank['ui']['h3_mask_repair'][0]['mask_pixels']==[0]*4
 assert m.H3MaskRepair().check_lazy_status(images)==[]
 click_spec=json.dumps({'signature':blank['ui']['h3_mask_repair'][0]['signature'],'steps':[step]})
 assert m.H3MaskRepair().check_lazy_status(images,repairs=click_spec)==['model']
 paint_spec=json.dumps({'signature':blank['ui']['h3_mask_repair'][0]['signature'],'steps':[paint]})
 assert m.H3MaskRepair().repair(images,repairs=paint_spec)['result'][0][2,8,8]==1
 result=m.H3MaskRepair().repair(images,masks,visible_frames=4)
 assert result['ui']['h3_mask_repair'][0]['frames']==4
 assert len(list((Path(d)/result['ui']['h3_mask_repair'][0]['directory']).glob('frame-*.jpg')))==4
 old_repairs=json.dumps({'signature':'wrong','steps':[step]})
 assert m.H3MaskRepair().check_lazy_status(images,masks,True,old_repairs)==[]
 skipped=m.H3MaskRepair().repair(images,masks,repairs=old_repairs)
 assert torch.equal(skipped['result'][0],masks)
 assert skipped['ui']['h3_mask_repair'][0]['repairs_skipped']
 current=json.dumps({'signature':m.repair_signature(images,masks),'steps':[paint]})
 applied=m.H3MaskRepair().repair(images,masks,repairs=current)
 assert applied['result'][0][2,8,8]==1
 assert not applied['ui']['h3_mask_repair'][0]['repairs_skipped']
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

with tempfile.TemporaryDirectory() as d:
 m.folder_paths.get_temp_directory=lambda:d
 dirty=torch.ones_like(masks)
 replacement={**paint,'replace':True}
 clean=m.apply_mask_repairs(images,dirty,[replacement],segment,track)
 assert clean[2,0,0]==0 and clean[2,8,8]==1 and clean[0].all()
 empty=m.apply_mask_repairs(images,dirty,[{'frame':1,'end':3,'replace':True}],segment,track)
 assert not empty[1:4].any() and empty[0].all() and empty[4].all()
 tiny=torch.zeros(1,16,1600);tiny[0,2,3]=.0001
 preview=m.H3MaskRepair().repair(torch.zeros(1,16,1600,3),tiny)['ui']['h3_mask_repair'][0]
 assert preview['mask_pixels']==[1]
 from PIL import Image
 import numpy as np
 assert np.asarray(Image.open(Path(d)/preview['directory']/'mask-0.png')).any()
print('PASS: replace removes old islands, empty range stays empty, subpixel masks remain visible')

objects=torch.zeros(5,16,16);objects[:,2:6,2:6]=1;objects[:,10:14,10:14]=1
remove={'frame':1,'end':3,'erase_only':True,'erase_points':[{'x':.2,'y':.2}]}
remaining=m.apply_mask_repairs(images,objects,[remove],segment,track)
assert not remaining[1:4,2:6,2:6].any() and remaining[:,10:14,10:14].all()
assert torch.equal(remaining[0],objects[0]) and torch.equal(remaining[4],objects[4])
assert torch.all(remaining<=objects)
drag={'frame':0,'end':0,'erase_only':True,'strokes':[{'mode':'erase','width':.1,'points':[{'x':.2,'y':.2},{'x':.2,'y':.3}]}]}
remaining=m.apply_mask_repairs(images,objects,[drag],segment,track)
assert remaining[0].sum()<objects[0].sum() and remaining[0,10:14,10:14].all()
assert torch.equal(remaining[1:],objects[1:])
print('PASS: click erase removes selected island; forward erase only subtracts; drag preserves other frames')
