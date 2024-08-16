import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {infer,featuresFromLandmarks,StableGesture} from '../demo/model.mjs';
const data=JSON.parse(await readFile(new URL('../demo/replay.json',import.meta.url))),weights=JSON.parse(await readFile(new URL('../demo/model.json',import.meta.url)));
let maxError=0;
for(const frame of data.frames){const scores=infer(frame.values,weights);assert.equal(scores.length,5);assert.ok(Math.abs(scores.reduce((a,b)=>a+b,0)-1)<1e-12);scores.forEach((p,i)=>{maxError=Math.max(maxError,Math.abs(p-frame.scores[i]));});}
assert.ok(maxError<1e-5,`Model mismatch: ${maxError}`);
const points=Array.from({length:21},(_,i)=>({x:.25+i*.0125,y:.9-i*.025}));const features=featuresFromLandmarks(points,960,540);
assert.equal(features.length,42);assert.deepEqual(features.slice(0,2),[0,0]);assert.equal(Math.max(...features.map(Math.abs)),1);
assert.deepEqual(featuresFromLandmarks([],960,540),[]);
const latch=new StableGesture();assert.equal(latch.update(1,0),null);assert.equal(latch.update(1,349),null);assert.equal(latch.update(1,350),1);assert.equal(latch.update(1,1800),null);latch.update(null,2000);latch.update(null,2300);assert.equal(latch.update(1,2400),null);assert.equal(latch.update(1,2800),1);
console.log(`120 recorded samples match the saved-model reference (max error ${maxError.toExponential(2)}). Feature and stable-trigger checks passed.`);
