import {test} from 'node:test';
import assert from 'node:assert/strict';
import {estimate} from '../src/estimate.js';
const base={parameters:8,layers:32,width:4096,tokens:1e9,sequence:2048,epochs:1,count:1,tflops:312,utilization:40,scaling:90,overhead:0,rate:2,memory:80};
test('compute includes forward/backward and sequence-dependent attention',()=>{const r=estimate(base);assert.equal(r.flops,(6*8e9+12*32*4096*2048)*1e9);assert.equal(r.gpuHours,r.flops/(312e12*.4)/3600);assert.equal(r.cost,r.gpuHours*2);});
test('ideal scaling reduces elapsed time without reducing billed GPU hours',()=>{const one=estimate(base),eight=estimate({...base,count:8,scaling:100});assert.equal(one.gpuHours,eight.gpuHours);assert.equal(eight.wallHours,one.wallHours/8);});
test('cluster inefficiency, overhead, and epochs affect compute correctly',()=>{const one=estimate(base),eight=estimate({...base,count:8,overhead:10,epochs:2});assert.ok(Math.abs(eight.gpuHours/one.gpuHours-2*1.1/.9)<1e-10);});
test('memory check accounts for optimizer states and headroom',()=>{assert.equal(estimate(base).fits,false);assert.equal(estimate({...base,count:8}).fits,true);});
test('invalid or nonfinite inputs fail rather than yielding misleading numbers',()=>{for(const changes of [{count:0},{tokens:NaN},{parameters:-1},{rate:-1},{utilization:101},{count:1.5}])assert.throws(()=>estimate({...base,...changes}));});
