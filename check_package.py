import asyncio
import json
from pathlib import Path
import nodes

root = Path(__file__).resolve().parent
assert asyncio.run(nodes.load_custom_node(str(root)))
expected = {'H3SwapTimeline24', 'H3SwapPrepare', 'H3SwapFinish', 'H3VideoResolution', 'H3SAMTrackCache', 'H3MaskRepair', 'H3MaskFrameGate', 'SolAttnMiniMax'}
assert expected <= nodes.NODE_CLASS_MAPPINGS.keys()
for name in expected:
    assert nodes.NODE_CLASS_MAPPINGS[name].INPUT_TYPES()
for path in (root / 'workflows').glob('*.json'):
    workflow = json.loads(path.read_text())
    if 'TwoPeople' not in path.name and any(n['type'] == 'H3MaskRepair' for n in workflow['nodes']):
        single_nodes = {n['id']: n for n in workflow['nodes']}
        single_links = {link[0]: link for link in workflow['links']}
        assert single_nodes[175]['mode'] == 0
        crop_mask = next(i['link'] for i in single_nodes[181]['inputs'] if i['name'] == 'masks')
        assert single_links[crop_mask][1:3] == [175, 0]
        assert next(i['link'] for i in single_nodes[204]['inputs'] if i['name'] == 'cropped_masks') is None
    graphs = [workflow] + workflow.get('definitions', {}).get('subgraphs', [])
    for graph in graphs:
        for node in graph['nodes']:
            values = node.get('widgets_values')
            if node['type'] == 'UNETLoader':
                assert values[0] == 'Minimax-h3_Singularity_ref2va_pruned_v1.3_int8.safetensors'
            if node['type'] == 'Power Lora Loader (rgthree)':
                loras = [v for v in values if isinstance(v, dict) and 'lora' in v]
                assert any(v['on'] and v['strength'] == .6 and 'turbo_4step_v0.1' in v['lora'] for v in loras)
                assert any(not v['on'] and v['lora'] == 'natural_face_speech_h3_lora_v1_500.safetensors' for v in loras)
            if node['type'] == 'H3MaskRepair':
                assert json.loads(values[0]) == {}
    print('PASS', path.name)
print('PASS: eight node classes registered with ComfyUI')
