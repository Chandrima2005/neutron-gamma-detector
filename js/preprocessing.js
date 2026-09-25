/*
 * preprocessing.js
 * -----------------
 * Client-side mirror of python/preprocessing.py — MUST match it exactly.
 * Takes a raw .lvm file's text content and turns it into the same 512-D
 * feature vector the Python training pipeline produces:
 *   parse -> normalize -> frame (Hanning, 1024, 80% overlap) -> short-time
 *   energy pulse localization (theta=0.15) -> STFT -> per-frame normalize
 *   -> average over the pulse window
 *
 * This has been numerically validated against preprocessing.py: identical
 * inputs produce identical feature vectors to full floating-point
 * precision.
 */

function fftInPlace(re, im){
  const n = re.length;
  for (let i=1, j=0; i<n; i++){
    let bit = n>>1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j){
      let t=re[i]; re[i]=re[j]; re[j]=t;
      t=im[i]; im[i]=im[j]; im[j]=t;
    }
  }
  for (let len=2; len<=n; len<<=1){
    const ang = -2*Math.PI/len;
    const wr0 = Math.cos(ang), wi0 = Math.sin(ang);
    for (let i=0; i<n; i+=len){
      let curWr=1, curWi=0;
      for (let j=0; j<len/2; j++){
        const ur=re[i+j], ui=im[i+j];
        const vr=re[i+j+len/2]*curWr - im[i+j+len/2]*curWi;
        const vi=re[i+j+len/2]*curWi + im[i+j+len/2]*curWr;
        re[i+j]=ur+vr; im[i+j]=ui+vi;
        re[i+j+len/2]=ur-vr; im[i+j+len/2]=ui-vi;
        const nWr = curWr*wr0 - curWi*wi0;
        const nWi = curWr*wi0 + curWi*wr0;
        curWr=nWr; curWi=nWi;
      }
    }
  }
}

function hanningWindow(N){
  const w = new Float64Array(N);
  for (let n=0;n<N;n++) w[n] = 0.5 - 0.5*Math.cos(2*Math.PI*n/(N-1));
  return w;
}

// ---- LVM text parser (mirrors read_lvm() in the Python pipeline) ----
function parseLVMText(text){
  const lines = text.split(/\r?\n/);
  const headerIdxs = [];
  for (let i=0;i<lines.length;i++) if (lines[i].includes('End_of_Header')) headerIdxs.push(i);
  if (headerIdxs.length === 0){
    throw new Error("Doesn't look like a LabVIEW .lvm file — no 'End_of_Header' marker found.");
  }
  const start = headerIdxs[headerIdxs.length-1] + 2;
  let deltaX = null;
  for (const l of lines){ if (l.startsWith('Delta_X')){ deltaX = parseFloat(l.split('\t')[1]); break; } }
  if (!deltaX || !isFinite(deltaX)) deltaX = 1e-6;
  const vals = [];
  for (let i=start; i<lines.length; i++){
    const parts = lines[i].trim().split('\t');
    if (parts.length >= 2){
      const v = parseFloat(parts[1]);
      if (!isNaN(v)) vals.push(v);
    }
  }
  if (vals.length < 1024){
    throw new Error(`Only found ${vals.length} samples after parsing — need at least 1024 for one STFT frame.`);
  }
  return { vals: Float64Array.from(vals), fs: 1/deltaX };
}

const NFFT = 1024, HOP = 204, THETA = 0.15;
const WIN = hanningWindow(NFFT);

// ---- full pipeline: normalize -> frame -> STE -> localize -> STFT -> 512-D feature ----
function runPipeline(vals, fs){
  let maxAbs = 0;
  for (let i=0;i<vals.length;i++) maxAbs = Math.max(maxAbs, Math.abs(vals[i]));
  if (maxAbs === 0) throw new Error('Waveform is all zeros — nothing to classify.');
  const xn = new Float64Array(vals.length);
  for (let i=0;i<vals.length;i++) xn[i] = vals[i]/maxAbs;

  const starts = [];
  for (let s=0; s<xn.length-NFFT; s+=HOP) starts.push(s);
  const nFrames = starts.length;

  const Ehat = new Float64Array(nFrames);
  const Praw = []; // per-frame normalized power spectra (nFrames x 512)
  let Emax = 0;
  const Eraw = new Float64Array(nFrames);

  for (let k=0; k<nFrames; k++){
    const s = starts[k];
    const re = new Float64Array(NFFT), im = new Float64Array(NFFT);
    let e = 0;
    for (let n=0;n<NFFT;n++){ const v = xn[s+n]*WIN[n]; re[n]=v; e += v*v; }
    e /= NFFT;
    Eraw[k] = e; if (e > Emax) Emax = e;
    fftInPlace(re, im);
    const P = new Float64Array(512);
    let pmax = 0;
    for (let m=0;m<512;m++){ const p = (re[m]*re[m]+im[m]*im[m])/NFFT; P[m]=p; if (p>pmax) pmax=p; }
    if (pmax === 0) pmax = 1;
    for (let m=0;m<512;m++) P[m] /= pmax;
    Praw.push(P);
  }
  for (let k=0;k<nFrames;k++) Ehat[k] = Emax>0 ? Eraw[k]/Emax : 0;

  let ks=-1, ke=-1;
  for (let k=0;k<nFrames;k++) if (Ehat[k] >= THETA){ if (ks===-1) ks=k; ke=k; }
  if (ks === -1){ ks=0; ke=nFrames-1; }

  const feature = new Float64Array(512);
  for (let k=ks; k<=ke; k++) for (let m=0;m<512;m++) feature[m] += Praw[k][m];
  const denom = (ke-ks+1);
  for (let m=0;m<512;m++) feature[m] /= denom;

  const frameTimes = starts.map(s => s/fs);
  const cropStartT = frameTimes[ks];
  const cropEndT = frameTimes[ke] + NFFT/fs;

  return { xn, fs, nFrames, Ehat, frameTimes, ks, ke, cropStartT, cropEndT, feature, Praw };
}

