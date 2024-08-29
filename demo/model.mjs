/** Forward pass through the saved course MLP (42 → 20 → 10 → 5). */
export function infer(values,weights){
 if(values.length!==42||!values.every(Number.isFinite))throw new Error('Expected 42 finite landmark features.');
 let x=values;
 for(let layer=0;layer<3;layer++){
  const W=weights['W'+layer],b=weights['b'+layer];
  x=b.map((bias,j)=>x.reduce((sum,v,i)=>sum+v*W[i][j],bias));
  if(layer<2)x=x.map(v=>Math.max(0,v));
 }
 const max=Math.max(...x),exp=x.map(v=>Math.exp(v-max)),total=exp.reduce((a,b)=>a+b,0);
 return exp.map(v=>v/total);
}
/** The course used mirrored, integer pixel coordinates before wrist normalization. */
export function featuresFromLandmarks(points,width,height){
 if(points.length!==21)return [];
 const pixels=points.map(p=>[Math.max(0,Math.min(Math.floor((1-p.x)*width),width-1)),Math.max(0,Math.min(Math.floor(p.y*height),height-1))]);
 const values=pixels.flatMap(p=>[p[0]-pixels[0][0],p[1]-pixels[0][1]]);
 const scale=Math.max(...values.map(Math.abs));return scale?values.map(v=>v/scale):Array(42).fill(0);
}
