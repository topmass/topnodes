import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// Run the editor's actual session switch without loading images or ComfyUI.
const source = readFileSync(new URL('./web/repair.js', import.meta.url), 'utf8');
const start = source.indexOf('  const info=node.repairPreview;', source.indexOf(' async function refresh()'));
const end = source.indexOf('  const token=++loadId;', start);
assert(start >= 0 && end > start);
const refresh = new (Object.getPrototypeOf(async function(){}).constructor)(
  'node', 'spec', 'widget', 'save', 'blank', 'queueRepair', 'status', source.slice(start, end));
for (const editor_first of [false, true]) {
  const oldSteps = [{frame: 0, end: 2, replace: true}];
  const widget = {value: JSON.stringify({signature: 'old', steps: oldSteps})};
  const node = {properties: {editor_first}, repairPreview: {signature: 'new'}};
  const spec = () => JSON.parse(widget.value);
  let queued = 0;
  const run = () => refresh(node, spec, widget, () => {}, () => {}, () => {queued++;}, {});
  await run();
  assert.equal(spec().signature, 'new');
  assert.deepEqual(spec().steps, []);
  assert.deepEqual(spec().sessions.old, oldSteps);
  node.repairPreview.signature = 'old';
  await run();
  assert.deepEqual(spec().steps, oldSteps);
  assert.equal(queued, 1);
  await run();
  assert.equal(queued, 1);
}
console.log('PASS: both editors switch mask sessions, preserve old edits, and replay restored edits once');
