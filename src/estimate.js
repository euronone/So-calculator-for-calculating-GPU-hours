export const models = [
 {id:'llama8',name:'Llama 3.1',family:'Meta',size:8,layers:32,width:4096,label:'8B'},
 {id:'llama70',name:'Llama 3.1',family:'Meta',size:70,layers:80,width:8192,label:'70B'},
 {id:'qwen7',name:'Qwen 2.5',family:'Alibaba',size:7.61,layers:28,width:3584,label:'7B'},
 {id:'qwen32',name:'Qwen 2.5',family:'Alibaba',size:32.5,layers:64,width:5120,label:'32B'},
 {id:'mistral',name:'Mistral v0.3',family:'Mistral AI',size:7.25,layers:32,width:4096,label:'7B'},
 {id:'gemma',name:'Gemma 2',family:'Google',size:9.24,layers:42,width:3584,label:'9B'},
 {id:'phi',name:'Phi-3 mini',family:'Microsoft',size:3.82,layers:32,width:3072,label:'3.8B'},
 {id:'custom',name:'Custom dense model',family:'Your architecture',size:13,layers:40,width:5120,label:'Custom'}
];
export const gpus = [
 {id:'h100',name:'NVIDIA H100',variant:'SXM · 80 GB',memory:80,tflops:989,rate:3.5},
 {id:'a100',name:'NVIDIA A100',variant:'SXM · 80 GB',memory:80,tflops:312,rate:1.8},
 {id:'a10040',name:'NVIDIA A100',variant:'SXM · 40 GB',memory:40,tflops:312,rate:1.4},
 {id:'l40s',name:'NVIDIA L40S',variant:'48 GB',memory:48,tflops:362,rate:1.2},
 {id:'4090',name:'NVIDIA RTX 4090',variant:'24 GB',memory:24,tflops:165,rate:0.5}
];
export function estimate({parameters,layers,width,tokens,sequence,epochs,count,tflops,utilization,scaling,overhead,rate,memory}) {
 const values=[parameters,layers,width,tokens,sequence,epochs,count,tflops,utilization,scaling,memory];
 if(values.some(v=>!Number.isFinite(v)||v<=0)||!Number.isFinite(rate)||rate<0||!Number.isFinite(overhead)||overhead<0||utilization>100||scaling>100||!Number.isInteger(count)) throw new Error('Enter valid positive training inputs.');
 const totalTokens=tokens*epochs;
 const flops=(6*parameters*1e9+12*layers*width*sequence)*totalTokens;
 const effective=tflops*1e12*(utilization/100)*(count===1?1:scaling/100);
 const gpuHours=flops/effective/3600*(1+overhead/100);
 const wallHours=gpuHours/count;
 // Mixed precision Adam: weights, gradients, master weights and optimizer states.
 const stateGB=parameters*16;
 const capacityGB=count*memory;
 return {totalTokens,flops,gpuHours,wallHours,cost:gpuHours*rate,stateGB,capacityGB,fits:stateGB<=capacityGB*.8,low:gpuHours*.75,high:gpuHours*1.5};
}
