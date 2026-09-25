/*
 * app.js
 * ------
 * UI logic only: renders the sample-event gallery (from sample_data.js),
 * the distribution/loss/confusion-matrix charts, and wires up the file
 * upload -> preprocessing.js (signal processing) -> model.js (inference)
 * -> on-screen result flow for the Live Classifier section.
 *
 * Load order (see index.html): sample_data.js, model_weights.js,
 * preprocessing.js, model.js, then this file.
 */

const css = getComputedStyle(document.documentElement);
const COL = {
  neutron: css.getPropertyValue('--neutron').trim(),
  gamma: css.getPropertyValue('--gamma').trim(),
  text: css.getPropertyValue('--text').trim(),
  dim: css.getPropertyValue('--text-dim').trim(),
  faint: css.getPropertyValue('--text-faint').trim(),
  line: css.getPropertyValue('--line').trim(),
  amber: css.getPropertyValue('--amber').trim(),
  good: css.getPropertyValue('--good').trim(),
  bad: css.getPropertyValue('--bad').trim(),
};

function dprCanvas(canvas){
  const dpr = Math.min(window.devicePixelRatio||1, 2);
  const w = canvas.clientWidth || canvas.width;
  const h = canvas.clientHeight || canvas.height;
  canvas.width = w*dpr; canvas.height = h*dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr,0,0,dpr,0,0);
  return {ctx, w, h};
}

function drawGridBg(ctx,w,h,padL,padR,padT,padB,rows=4){
  ctx.strokeStyle = 'rgba(140,200,230,0.07)';
  ctx.lineWidth = 1;
  for(let i=0;i<=rows;i++){
    const y = padT + (h-padT-padB)*i/rows;
    ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(w-padR,y); ctx.stroke();
  }
}

function lineChart(canvas, series, opts={}){
  const {ctx,w,h} = dprCanvas(canvas);
  ctx.clearRect(0,0,w,h);
  const padL=opts.padL??8, padR=opts.padR??8, padT=opts.padT??10, padB=opts.padB??22;
  drawGridBg(ctx,w,h,padL,padR,padT,padB, opts.rows||4);
  let allX=[],allY=[];
  series.forEach(s=>s.points.forEach(p=>{allX.push(p[0]);allY.push(p[1]);}));
  const xmin=opts.xmin??Math.min(...allX), xmax=opts.xmax??Math.max(...allX);
  const ymin=opts.ymin??0, ymax=opts.ymax??Math.max(...allY)*1.08;
  const X = x => padL + (w-padL-padR)*(x-xmin)/(xmax-xmin||1);
  const Y = y => h-padB - (h-padT-padB)*(y-ymin)/(ymax-ymin||1);
  series.forEach(s=>{
    ctx.beginPath();
    s.points.forEach((p,i)=>{ i===0?ctx.moveTo(X(p[0]),Y(p[1])):ctx.lineTo(X(p[0]),Y(p[1])); });
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width||1.8;
    ctx.shadowColor = s.color; ctx.shadowBlur = s.glow||0;
    ctx.stroke();
    ctx.shadowBlur = 0;
  });
  // axis labels
  ctx.fillStyle = COL.faint; ctx.font = '10px "IBM Plex Mono",monospace';
  if(opts.xLabel) ctx.fillText(opts.xLabel, padL, h-6);
  if(opts.xmaxLabel) { ctx.textAlign='right'; ctx.fillText(opts.xmaxLabel, w-padR, h-6); ctx.textAlign='left'; }
  return {X,Y,padL,padR,padT,padB,w,h};
}

/* ---------------- HERO SIGNALS (neutron + gamma side by side) ---------------- */
const REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const curatedOk = cls => SAMPLE_DATA.curated_events.filter(e => e.class===cls && e.lr_pred===(cls==='Neutron'?1:0));
const HERO = [
  { canvasId:'heroNeutron', fileId:'heroNeutronFile', cardId:'heroNeutronCard', color:COL.neutron,
    evt: SAMPLE_DATA.curated_events.find(e=>e.file==='scope-1_026.lvm' && e.class==='Neutron') || curatedOk('Neutron')[0] },
  { canvasId:'heroGamma', fileId:'heroGammaFile', cardId:'heroGammaCard', color:COL.gamma,
    evt: curatedOk('Gamma')[0] || SAMPLE_DATA.curated_events.find(e=>e.class==='Gamma') },
];

// progress 0..1: how much of the trace is drawn (for the sweep-in animation)
function drawHeroSignal(h, progress=1){
  const e = h.evt, canvas = document.getElementById(h.canvasId);
  const all = e.waveform.t_ms.map((t,i)=>[t, e.waveform.amp[i]]);
  const n = Math.max(2, Math.round(all.length*progress));
  const xmin = all[0][0], xmax = all[all.length-1][0];
  const {X,Y,padT,padB} = lineChart(canvas,[{points:all.slice(0,n),color:h.color,width:1.4,glow:8}],
    {xmin,xmax,ymin:-1.05,ymax:1.05,padT:10,padB:20,xLabel:'0 ms',xmaxLabel:'20 ms'});
  const ctx = canvas.getContext('2d');
  const x0=X(e.crop_ms[0]), x1=X(e.crop_ms[1]);
  ctx.fillStyle = h.color + '1f';   // ~12% alpha of the class colour
  ctx.fillRect(x0,padT,x1-x0,canvas.clientHeight-padT-padB);
  // leading dot while sweeping
  if (progress < 1){
    const p = all[n-1];
    ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3, 0, Math.PI*2);
    ctx.fillStyle = h.color; ctx.shadowColor = h.color; ctx.shadowBlur = 12; ctx.fill(); ctx.shadowBlur = 0;
  }
}

function animateHeroSignal(h, duration=1600){
  if (REDUCED_MOTION){ drawHeroSignal(h,1); return; }
  const token = h.anim = {};
  const t0 = performance.now();
  const step = now => {
    if (h.anim !== token) return;               // a newer animation took over
    const p = Math.min(1, (now-t0)/duration);
    drawHeroSignal(h, 1-Math.pow(1-p,3));       // ease-out
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function initHero(){
  HERO.forEach(h=>{
    document.getElementById(h.fileId).textContent = h.evt.file;
    const card = document.getElementById(h.cardId);
    card.addEventListener('mouseenter', ()=>animateHeroSignal(h, 1100));
    animateHeroSignal(h);
  });
}
function drawHero(){ HERO.forEach(h=>{ h.anim = null; drawHeroSignal(h,1); }); }

/* ---------------- PIPELINE STEPS ---------------- */
const steps = [
 ["01","Acquire","20 ms .lvm capture per event, 4 ms pre-trigger, 1 MSa/s LabVIEW DAQ."],
 ["02","Normalize","Amplitude-normalize each waveform by its own peak."],
 ["03","Localize","1024-sample Hanning frames, 80% overlap → short-time energy → threshold crosses mark the pulse window."],
 ["04","STFT","1024-pt FFT per frame inside the pulse window, per-frame spectrum normalized."],
 ["05","Average","Temporal mean of normalized spectra → 512-D feature vector per event."],
 ["06","Classify","Logistic Regression & Linear SVM, SGD-trained, 30k epochs, 7:2:1 split."],
];
document.getElementById('pipelineSteps').innerHTML = steps.map(s=>`
  <div class="pstep glow-card reveal"><div class="pn">${s[0]}</div><h4>${s[1]}</h4><p>${s[2]}</p><span class="arrow"></span></div>
`).join('');

/* ---------------- DISTRIBUTION CHART ---------------- */
let distMode = 'train';
function drawDist(){
  const d = SAMPLE_DATA.distribution[distMode];
  const freq = SAMPLE_DATA.distribution.freq_khz;
  const nPts = freq.map((f,i)=>[f,d.neutron[i]]);
  const gPts = freq.map((f,i)=>[f,d.gamma[i]]);
  lineChart(document.getElementById('distChart'),
    [{points:gPts,color:COL.gamma,width:1.6},{points:nPts,color:COL.neutron,width:1.6}],
    {ymin:0,ymax:1.02,padL:6,padR:6,padT:10,padB:22,xLabel:'0 kHz',xmaxLabel:'~500 kHz'});
  document.getElementById('distToggleLabel').textContent = (distMode==='train'?'TRAIN SET':'TEST SET');
}
document.getElementById('distToggleBtn').addEventListener('click',()=>{
  distMode = distMode==='train'?'test':'train';
  document.getElementById('distToggleBtn').textContent = distMode==='train'? 'Show test set →':'Show train set →';
  drawDist();
});

/* ---------------- EXPLORER ---------------- */
const listEl = document.getElementById('eventList');
SAMPLE_DATA.curated_events.forEach((e,i)=>{
  const btn = document.createElement('button');
  btn.className='event-item'; btn.dataset.id=e.id;
  btn.innerHTML = `<span>${e.file.replace('scope-1_','#')}</span><span class="cls ${e.class.toLowerCase()}">${e.class[0]}</span>`;
  btn.addEventListener('click', ()=>selectEvent(e.id));
  listEl.appendChild(btn);
});

function heat(v){
  // 0..1 -> phosphor colormap black -> amber -> white
  v = Math.max(0,Math.min(1,v));
  const stops = [
    [0.00, [5,8,10]],
    [0.35, [90,40,10]],
    [0.65, [255,150,30]],
    [1.00, [255,240,200]],
  ];
  for(let i=0;i<stops.length-1;i++){
    const [t0,c0]=stops[i], [t1,c1]=stops[i+1];
    if(v>=t0 && v<=t1){
      const f=(v-t0)/(t1-t0);
      return `rgb(${c0[0]+f*(c1[0]-c0[0])|0},${c0[1]+f*(c1[1]-c0[1])|0},${c0[2]+f*(c1[2]-c0[2])|0})`;
    }
  }
  return 'rgb(255,240,200)';
}

function selectEvent(id){
  document.querySelectorAll('.event-item').forEach(b=>b.classList.toggle('active', +b.dataset.id===id));
  const e = SAMPLE_DATA.curated_events.find(ev=>ev.id===id);
  const panel = document.querySelector('#explorer .scope-panel');
  panel.classList.toggle('is-neutron', e.class==='Neutron');
  panel.classList.toggle('is-gamma', e.class==='Gamma');

  // waveform
  const wavePts = e.waveform.t_ms.map((t,i)=>[t,e.waveform.amp[i]]);
  const col = e.class==='Neutron'?COL.neutron:COL.gamma;
  const wc = document.getElementById('waveCanvas');
  const {X,Y,padT,padB} = lineChart(wc,[{points:wavePts,color:col,width:1.3}],
    {ymin:-1.05,ymax:1.05,padL:6,padR:6,padT:8,padB:16});
  { const ctx=wc.getContext('2d'); const dpr=Math.min(window.devicePixelRatio||1,2); ctx.setTransform(dpr,0,0,dpr,0,0);
    const x0=X(e.crop_ms[0]), x1=X(e.crop_ms[1]);
    ctx.fillStyle='rgba(255,194,51,0.12)'; ctx.fillRect(x0,padT,x1-x0,wc.clientHeight-padT-padB); }
  document.getElementById('cropLabel').textContent = `crop ${e.crop_ms[0].toFixed(2)}–${e.crop_ms[1].toFixed(2)} ms`;

  // STE
  const stePts = e.ste.t_ms.map((t,i)=>[t,e.ste.e[i]]);
  const sc = document.getElementById('steCanvas');
  const r2 = lineChart(sc,[{points:stePts,color:COL.amber,width:1.3}],
    {ymin:0,ymax:1.05,padL:6,padR:6,padT:8,padB:16});
  { const ctx=sc.getContext('2d'); const dpr=Math.min(window.devicePixelRatio||1,2); ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.strokeStyle='rgba(255,107,122,0.6)'; ctx.setLineDash([4,3]); ctx.lineWidth=1;
    const yTh = r2.Y(0.15);
    ctx.beginPath(); ctx.moveTo(r2.padL,yTh); ctx.lineTo(sc.clientWidth-r2.padR,yTh); ctx.stroke();
    ctx.setLineDash([]); }

  // spectrogram
  const spc = document.getElementById('specCanvas');
  const {ctx:sctx,w:sw,h:sh} = dprCanvas(spc);
  sctx.clearRect(0,0,sw,sh);
  const grid = e.spectrogram; // t_out x f_out
  const T = grid.length, F = grid[0].length;
  const sPadL=36, sPadR=10, sPadT=8, sPadB=18;
  const cw=(sw-sPadL-sPadR)/T, ch=(sh-sPadT-sPadB)/F;
  for(let ti=0; ti<T; ti++){
    for(let fi=0; fi<F; fi++){
      sctx.fillStyle = heat(grid[ti][fi]);
      sctx.fillRect(sPadL+ti*cw, sh-sPadB-(fi+1)*ch, cw+0.6, ch+0.6);
    }
  }
  sctx.fillStyle = COL.faint; sctx.font='10px "IBM Plex Mono",monospace';
  sctx.fillText('0',sPadL-2,sh-sPadB+12);
  sctx.fillText('500 kHz',sPadL-2,sPadT+8);
  sctx.fillText(e.crop_ms[0].toFixed(2)+' ms',sPadL,sh-4);
  sctx.textAlign='right'; sctx.fillText(e.crop_ms[1].toFixed(2)+' ms', sw-sPadR, sh-4); sctx.textAlign='left';

  // predictions
  const lrPredCls = e.lr_pred===1?'Neutron':'Gamma';
  const svmPredCls = e.svm_pred===1?'Neutron':'Gamma';
  document.getElementById('lrOut').innerHTML = `<span class="badge ${lrPredCls.toLowerCase()}"></span>${lrPredCls}`;
  document.getElementById('lrConf').textContent = `P(neutron) = ${e.lr_proba.toFixed(3)}`;
  document.getElementById('svmOut').innerHTML = `<span class="badge ${svmPredCls.toLowerCase()}"></span>${svmPredCls}`;
  document.getElementById('svmConf').textContent = `decision score = ${e.svm_score.toFixed(3)}`;

  const lrOk = (lrPredCls===e.class), svmOk = (svmPredCls===e.class);
  document.getElementById('truthLine').innerHTML =
    `Ground truth: <b>${e.class}</b> (${e.file}) &nbsp;·&nbsp; LR <span class="${lrOk?'ok':'bad'}">${lrOk?'✓ correct':'✗ misclassified'}</span> &nbsp;·&nbsp; SVM <span class="${svmOk?'ok':'bad'}">${svmOk?'✓ correct':'✗ misclassified'}</span>`;
}

/* ---------------- LOSS CHART ---------------- */
function drawLoss(){
  const lr = SAMPLE_DATA.loss_curves.lr.map(p=>[p[0],p[1]]);
  const svm = SAMPLE_DATA.loss_curves.svm.map(p=>[p[0],p[1]]);
  lineChart(document.getElementById('lossChart'),
    [{points:lr,color:COL.neutron,width:1.8},{points:svm,color:COL.gamma,width:1.8}],
    {ymin:0,padL:8,padR:8,padT:10,padB:22,xLabel:'0',xmaxLabel:SAMPLE_DATA.metrics.epochs.toLocaleString()+' epochs'});
}

/* ---------------- CONFUSION MATRICES ---------------- */
function cmTable(elId, cm){
  // cm rows/cols: [Neutron, Gamma] (actual rows, predicted cols)
  const rowTotal = [cm[0][0]+cm[0][1], cm[1][0]+cm[1][1]];
  const cell = (v,rowIdx,isDiag)=>{
    const pct = rowTotal[rowIdx]>0 ? (v/rowTotal[rowIdx]*100) : 0;
    const alpha = 0.15+0.55*(pct/100);
    const color = isDiag? `rgba(91,227,164,${alpha})` : `rgba(255,107,122,${alpha})`;
    return `<td style="background:${color}"><div class="cm-cell">${v}</div><div style="font-size:10px;color:var(--text-faint)">${pct.toFixed(1)}%</div></td>`;
  };
  document.getElementById(elId).innerHTML = `
    <tr><th></th><th>Pred. Neutron</th><th>Pred. Gamma</th></tr>
    <tr><th>Actual Neutron</th>${cell(cm[0][0],0,true)}${cell(cm[0][1],0,false)}</tr>
    <tr><th>Actual Gamma</th>${cell(cm[1][0],1,false)}${cell(cm[1][1],1,true)}</tr>
  `;
}

/* ---------------- METRICS TABLE ---------------- */
function metricsTable(){
  const m = SAMPLE_DATA.metrics;
  const rows = [
    ['Accuracy', (m.lr.accuracy*100).toFixed(2)+'%', (m.svm.accuracy*100).toFixed(2)+'%'],
    ['Precision — Neutron', m.lr.precision.Neutron.toFixed(3), m.svm.precision.Neutron.toFixed(3)],
    ['Precision — Gamma', m.lr.precision.Gamma.toFixed(3), m.svm.precision.Gamma.toFixed(3)],
    ['Recall — Neutron', m.lr.recall.Neutron.toFixed(3), m.svm.recall.Neutron.toFixed(3)],
    ['Recall — Gamma', m.lr.recall.Gamma.toFixed(3), m.svm.recall.Gamma.toFixed(3)],
    ['F1 — Neutron', m.lr.f1.Neutron.toFixed(3), m.svm.f1.Neutron.toFixed(3)],
    ['F1 — Gamma', m.lr.f1.Gamma.toFixed(3), m.svm.f1.Gamma.toFixed(3)],
    ['Loss function', 'log-loss', 'hinge-loss'],
    ['Initial learning rate η₀', m.lr.hyperparams.eta0, m.svm.hyperparams.eta0],
  ];
  document.getElementById('metricsTable').innerHTML =
    `<tr><th>Metric</th><th>Logistic Regression</th><th>Linear SVM</th></tr>` +
    rows.map(r=>`<tr><td>${r[0]}</td><td class="num">${r[1]}</td><td class="num">${r[2]}</td></tr>`).join('');
  document.getElementById('ntest').textContent = m.split.n_test;
}

/* ================= LIVE CLASSIFIER UI: wires preprocessing.js + model.js to the page ================= */

function downsampleArr(arr, nOut){
  const out = new Array(nOut);
  for (let i=0;i<nOut;i++) out[i] = arr[Math.min(arr.length-1, Math.round(i*(arr.length-1)/(nOut-1)))];
  return out;
}

function renderLiveResult(fname, vals, fs){
  const r = runPipeline(vals, fs);
  const pred = predictLive(r.feature);

  // waveform
  const tFull = []; for (let i=0;i<r.xn.length;i++) tFull.push(i/fs*1000);
  const wIdx = []; for (let i=0;i<500;i++) wIdx.push(Math.round(i*(r.xn.length-1)/499));
  const wavePts = wIdx.map(i => [tFull[i], r.xn[i]]);
  const wc = document.getElementById('liveWaveCanvas');
  const {X,Y,padT,padB} = lineChart(wc,[{points:wavePts,color:COL.amber,width:1.3}],
    {ymin:-1.05,ymax:1.05,padL:6,padR:6,padT:8,padB:16});
  { const ctx=wc.getContext('2d'); const dpr=Math.min(window.devicePixelRatio||1,2); ctx.setTransform(dpr,0,0,dpr,0,0);
    const x0=X(r.cropStartT*1000), x1=X(r.cropEndT*1000);
    ctx.fillStyle='rgba(255,194,51,0.12)'; ctx.fillRect(x0,padT,x1-x0,wc.clientHeight-padT-padB); }
  document.getElementById('liveCropLabel').textContent = `crop ${(r.cropStartT*1000).toFixed(2)}–${(r.cropEndT*1000).toFixed(2)} ms`;

  // STE
  const steT = downsampleArr(r.frameTimes.map(t=>t*1000), Math.min(200, r.frameTimes.length));
  const steE = downsampleArr(Array.from(r.Ehat), Math.min(200, r.Ehat.length));
  const stePts = steT.map((t,i)=>[t, steE[i]]);
  const sc = document.getElementById('liveSteCanvas');
  const r2 = lineChart(sc,[{points:stePts,color:COL.gamma,width:1.3}], {ymin:0,ymax:1.05,padL:6,padR:6,padT:8,padB:16});
  { const ctx=sc.getContext('2d'); const dpr=Math.min(window.devicePixelRatio||1,2); ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.strokeStyle='rgba(255,107,122,0.6)'; ctx.setLineDash([4,3]); ctx.lineWidth=1;
    const yTh = r2.Y(0.15);
    ctx.beginPath(); ctx.moveTo(r2.padL,yTh); ctx.lineTo(sc.clientWidth-r2.padR,yTh); ctx.stroke();
    ctx.setLineDash([]); }

  // spectrogram (downsample nFrames x 512 -> 40 x 64)
  const T_OUT = 40, F_OUT = 64;
  const tIdx = []; for (let i=0;i<T_OUT;i++) tIdx.push(Math.round(i*(r.nFrames-1)/(T_OUT-1)));
  const spc = document.getElementById('liveSpecCanvas');
  const {ctx:sctx,w:sw,h:sh} = dprCanvas(spc);
  sctx.clearRect(0,0,sw,sh);
  const sPadL=36, sPadR=10, sPadT=8, sPadB=18;
  const cw=(sw-sPadL-sPadR)/T_OUT, ch=(sh-sPadT-sPadB)/F_OUT;
  for (let ti=0; ti<T_OUT; ti++){
    const P = r.Praw[tIdx[ti]];
    for (let fi=0; fi<F_OUT; fi++){
      let sum=0, cnt=0;
      const lo = Math.floor(fi*512/F_OUT), hi = Math.floor((fi+1)*512/F_OUT);
      for (let m=lo; m<Math.max(hi,lo+1); m++){ sum+=P[m]; cnt++; }
      sctx.fillStyle = heat(sum/cnt);
      sctx.fillRect(sPadL+ti*cw, sh-sPadB-(fi+1)*ch, cw+0.6, ch+0.6);
    }
  }
  sctx.fillStyle = COL.faint; sctx.font='10px "IBM Plex Mono",monospace';
  sctx.fillText('0',sPadL-2,sh-sPadB+12);
  sctx.fillText('500 kHz',sPadL-2,sPadT+8);
  sctx.fillText((r.cropStartT*1000).toFixed(2)+' ms',sPadL,sh-4);
  sctx.textAlign='right'; sctx.fillText((r.cropEndT*1000).toFixed(2)+' ms', sw-sPadR, sh-4); sctx.textAlign='left';

  // predictions
  const lrCls = pred.lrPred===1?'Neutron':'Gamma', svmCls = pred.svmPred===1?'Neutron':'Gamma';
  document.getElementById('liveLrOut').innerHTML = `<span class="badge ${lrCls.toLowerCase()}"></span>${lrCls}`;
  document.getElementById('liveLrConf').textContent = `P(neutron) = ${pred.lrProba.toFixed(3)}`;
  document.getElementById('liveSvmOut').innerHTML = `<span class="badge ${svmCls.toLowerCase()}"></span>${svmCls}`;
  document.getElementById('liveSvmConf').textContent = `decision score = ${pred.svmScore.toFixed(3)}`;
  document.getElementById('liveFileName').textContent = fname;
  document.getElementById('liveResults').style.display = 'block';

  const truthSel = document.getElementById('liveTruthSelect');
  truthSel.onchange = () => {
    const t = truthSel.value;
    const note = document.getElementById('liveAccuracyNote');
    if (!t){ note.textContent=''; return; }
    const lrOk = lrCls===t, svmOk = svmCls===t;
    note.innerHTML = `&nbsp;·&nbsp; LR <span class="${lrOk?'ok':'bad'}">${lrOk?'✓ correct':'✗ wrong'}</span> &nbsp;·&nbsp; SVM <span class="${svmOk?'ok':'bad'}">${svmOk?'✓ correct':'✗ wrong'}</span>`;
  };
  truthSel.value = ''; document.getElementById('liveAccuracyNote').textContent='';
}

function handleLiveFile(file){
  const errBox = document.getElementById('liveError');
  errBox.style.display = 'none';
  const reader = new FileReader();
  reader.onload = (e) => {
    try{
      const { vals, fs } = parseLVMText(e.target.result);
      renderLiveResult(file.name, vals, fs);
    } catch(err){
      errBox.textContent = 'Could not process this file: ' + err.message;
      errBox.style.display = 'block';
      document.getElementById('liveResults').style.display = 'none';
    }
  };
  reader.onerror = () => { errBox.textContent = 'Could not read this file from disk.'; errBox.style.display='block'; };
  reader.readAsText(file);
}

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
dropZone.addEventListener('click', ()=> fileInput.click());
fileInput.addEventListener('change', (e)=>{ if (e.target.files[0]) handleLiveFile(e.target.files[0]); });
['dragover','dragenter'].forEach(ev => dropZone.addEventListener(ev, (e)=>{ e.preventDefault(); dropZone.classList.add('drag'); }));
['dragleave','drop'].forEach(ev => dropZone.addEventListener(ev, (e)=>{ e.preventDefault(); dropZone.classList.remove('drag'); }));
dropZone.addEventListener('drop', (e)=>{ const f = e.dataTransfer.files[0]; if (f) handleLiveFile(f); });

/* ---------------- HOVER SPOTLIGHT, SCROLL REVEAL, COUNT-UP ---------------- */
document.addEventListener('pointermove', (ev)=>{
  const card = ev.target.closest && ev.target.closest('.glow-card');
  if (!card) return;
  const r = card.getBoundingClientRect();
  card.style.setProperty('--mx', (ev.clientX-r.left)+'px');
  card.style.setProperty('--my', (ev.clientY-r.top)+'px');
});

function countUp(el){
  const target = +el.dataset.count, suffix = el.dataset.suffix || '';
  if (REDUCED_MOTION){ el.textContent = target.toLocaleString()+suffix; return; }
  const t0 = performance.now(), dur = 1400;
  const step = now => {
    const p = Math.min(1,(now-t0)/dur);
    el.textContent = Math.round(target*(1-Math.pow(1-p,3))).toLocaleString()+suffix;
    if (p<1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function initReveal(){
  const els = document.querySelectorAll('.reveal');
  if (!('IntersectionObserver' in window)){ els.forEach(el=>el.classList.add('in')); return; }
  const io = new IntersectionObserver(entries=>{
    entries.forEach(en=>{
      if (!en.isIntersecting) return;
      const el = en.target;
      // stagger siblings that enter together
      const sibs = [...el.parentElement.children].filter(c=>c.classList.contains('reveal'));
      el.style.transitionDelay = (Math.max(0, sibs.indexOf(el))*70)+'ms';
      el.classList.add('in');
      setTimeout(()=>{ el.style.transitionDelay = ''; }, 1400);  // so hover isn't delayed later
      el.querySelectorAll('[data-count]').forEach(countUp);
      if (el.matches('[data-count]')) countUp(el);
      io.unobserve(el);
    });
  }, {threshold:0.12});
  els.forEach(el=>io.observe(el));
}

/* ---------------- INIT ---------------- */
function renderAll(){
  drawHero();
  drawDist();
  cmTable('cmLR', SAMPLE_DATA.metrics.lr.confusion_matrix);
  cmTable('cmSVM', SAMPLE_DATA.metrics.svm.confusion_matrix);
  metricsTable();
  drawLoss();
  const active = document.querySelector('.event-item.active');
  selectEvent(active ? +active.dataset.id : SAMPLE_DATA.curated_events[0].id);
}
renderAll();
initHero();
initReveal();
let resizeTimer;
window.addEventListener('resize', ()=>{ clearTimeout(resizeTimer); resizeTimer = setTimeout(renderAll, 150); });
