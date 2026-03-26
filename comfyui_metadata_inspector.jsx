import { useState, useRef, useCallback, useEffect } from "react";

// ── PNG parsing ──────────────────────────────────────────────────────────────
function dec(arr, s, e) {
  let r = "", e2 = e !== undefined ? e : arr.length;
  for (let i = s; i < e2; i++) r += String.fromCharCode(arr[i]);
  return r;
}
function parsePNG(buf) {
  const b = new Uint8Array(buf), sig = [137,80,78,71,13,10,26,10];
  for (let i = 0; i < 8; i++) if (b[i] !== sig[i]) throw new Error("Not PNG");
  const chunks = []; let p = 8;
  while (p + 8 < b.length) {
    const l = (b[p]<<24|b[p+1]<<16|b[p+2]<<8|b[p+3])>>>0; p += 4;
    const t = String.fromCharCode(b[p],b[p+1],b[p+2],b[p+3]); p += 4;
    if (t === "tEXt" || t === "iTXt") {
      const d = b.slice(p, p+l); let n = -1;
      for (let j = 0; j < d.length; j++) if (d[j]===0){n=j;break;}
      const kw = n>=0 ? dec(d,0,n) : "";
      let tx = "";
      if (t === "tEXt") { tx = n>=0 ? dec(d,n+1) : dec(d,0); }
      else {
        let q=n+1+2, le=-1;
        for (let x=q;x<d.length;x++) if(d[x]===0){le=x;break;}
        q=(le>=0?le:q)+1; let te=-1;
        for (let x=q;x<d.length;x++) if(d[x]===0){te=x;break;}
        q=(te>=0?te:q)+1; tx=dec(d,q);
      }
      chunks.push({ kw: kw.toLowerCase(), txt: tx });
    } else if (t==="IEND") break;
    p += l + 4;
  }
  return chunks;
}
function extractComfy(chunks) {
  let wf=null, pr=null;
  for (const c of chunks) {
    if (c.kw==="workflow"&&c.txt[0]==="{") { try{wf=JSON.parse(c.txt);}catch(e){} }
    if (c.kw==="prompt"&&c.txt[0]==="{") { try{pr=JSON.parse(c.txt);}catch(e){} }
    if (wf&&pr) break;
  }
  return (wf||pr) ? {workflow:wf,prompt:pr} : null;
}

// ── Metadata extraction ──────────────────────────────────────────────────────
const TE = ["CLIPTextEncode","CLIPTextEncodeSDXL","CLIPTextEncodeSDXLRefiner","CLIPTextEncodeFlux","CLIPTextEncodeSD3","smZ CLIPTextEncode","BNK_CLIPTextEncodeAdvanced","CLIPTextEncodeWithWeight","easy positive","easy negative","ttN text","CR Text","String Literal","StringLiteral"];
const LT = ["LoraLoader","LoraLoaderModelOnly","LoRALoader","LoraLoaderStack","CR LoRA Stack","LoraTagLoader","Power Lora Loader (rgthree)","LoraLoaderModelAndCLIP"];
const NH = ["worst quality","bad quality","lowres","blurry","nsfw","ugly","deformed","watermark","bad anatomy","bad hands","low quality","normal quality"];
const isTE = c => TE.includes(c);
const isLora = c => LT.includes(c) || c.toLowerCase().includes("lora");
const sn = s => { if(!s) return ""; const p=s.replace(/\\/g,"/").split("/"); return p[p.length-1].replace(/\.(safetensors|pt|ckpt|bin|gguf)$/i,""); };
const looksNeg = t => { const l=t.toLowerCase(); return NH.filter(h=>l.includes(h)).length>=2; };

function extractParams(row, wf, pr) {
  const lm={}, tn={};
  if (pr) {
    for (const id in pr) {
      if (!pr.hasOwnProperty(id)) continue;
      const n=pr[id], inp=n.inputs||{}, ct=n.class_type||"";
      if (ct==="KSampler"||ct==="KSamplerAdvanced") {
        if (inp.seed!==undefined) row.seed=inp.seed;
        if (inp.steps!==undefined) row.steps=inp.steps;
        if (inp.cfg!==undefined) row.cfg=parseFloat(inp.cfg).toFixed(1);
        if (inp.sampler_name) row.sampler=inp.sampler_name;
        if (inp.scheduler) row.scheduler=inp.scheduler;
        if (inp.denoise!==undefined) row.denoise=parseFloat(inp.denoise).toFixed(2);
      }
      if ((ct==="KSamplerSelect"||ct==="SamplerCustomAdvanced"||ct==="SamplerCustom")&&inp.sampler_name&&!row.sampler) row.sampler=inp.sampler_name;
      if ((ct==="BasicScheduler"||ct==="KarrasScheduler")) {
        if (inp.scheduler&&!row.scheduler) row.scheduler=inp.scheduler;
        if (inp.steps&&!row.steps) row.steps=inp.steps;
        if (inp.denoise!==undefined&&!row.denoise) row.denoise=parseFloat(inp.denoise).toFixed(2);
      }
      if (ct==="FluxGuidance"&&inp.guidance!==undefined&&!row.cfg) row.cfg=parseFloat(inp.guidance).toFixed(1);
      if (ct==="CheckpointLoaderSimple"||ct==="CheckpointLoader"||ct==="CheckpointLoaderNF4") { if(inp.ckpt_name&&!row.model) row.model=sn(inp.ckpt_name); }
      if (ct==="UNETLoader"||ct==="DiffusionModelLoader"||ct==="UnetLoaderGGUF") { if(inp.unet_name&&!row.model) row.model=sn(inp.unet_name); }
      if (isLora(ct)) {
        if (inp.lora_name) { const ln=sn(inp.lora_name); lm[ln]={name:ln,strength:inp.strength_model!==undefined?parseFloat(inp.strength_model).toFixed(2):(inp.strength!==undefined?parseFloat(inp.strength).toFixed(2):"1.00")}; }
        for (const k in inp) { if(!inp.hasOwnProperty(k)) continue; if(/^lora_\d+$/.test(k)&&inp[k]&&inp[k].on&&inp[k].lora){const ln2=sn(inp[k].lora);lm[ln2]={name:ln2,strength:inp[k].strength!==undefined?parseFloat(inp[k].strength).toFixed(2):"1.00"};} }
        for (const ki in inp) { if(!inp.hasOwnProperty(ki)) continue; if(/^lora_name_\d+$/.test(ki)&&inp[ki]){const ln3=sn(inp[ki]);const sk=ki.replace("lora_name","model_weight");lm[ln3]={name:ln3,strength:inp[sk]!==undefined?parseFloat(inp[sk]).toFixed(2):"1.00"};} }
      }
      if (isTE(ct)) {
        let txt=inp.text||inp.text_g||inp.clip_l||inp.t5xxl||inp.positive||inp.negative||"";
        if (Array.isArray(txt)) txt=txt.join(" ");
        txt=String(txt).trim();
        const re=/<lora:([^:>]+)(?::([0-9.]+))?>/g; let m;
        while((m=re.exec(txt))!==null){const lnt=sn(m[1]);lm[lnt]={name:lnt,strength:m[2]?parseFloat(m[2]).toFixed(2):"1.00"};}
        txt=txt.replace(/<lora:[^>]+>/g,"").replace(/\s+/g," ").trim();
        if (txt) tn[id]={text:txt};
      }
    }
    const pi={}, ni={};
    for (const id2 in pr) {
      if (!pr.hasOwnProperty(id2)) continue;
      const n2=pr[id2], i2=n2.inputs||{}, ct2=n2.class_type||"";
      if (["KSampler","KSamplerAdvanced","SamplerCustomAdvanced","SamplerCustom","KSamplerEfficient"].includes(ct2)) {
        if (Array.isArray(i2.positive)) pi[String(i2.positive[0])]=true;
        if (Array.isArray(i2.negative)) ni[String(i2.negative[0])]=true;
      }
      if (ct2==="FluxGuidance"&&Array.isArray(i2.conditioning)) pi[String(i2.conditioning[0])]=true;
    }
    function tu(sid,v){if(!sid||v[sid])return;v[sid]=true;const nd=pr[sid];if(!nd)return;const i3=nd.inputs||{};for(const k3 in i3){if(!i3.hasOwnProperty(k3))continue;if(Array.isArray(i3[k3])&&i3[k3].length===2&&typeof i3[k3][0]==="number")tu(String(i3[k3][0]),v);}}
    const pv={},nv={};
    for (const pp in pi) tu(pp,pv);
    for (const np in ni) tu(np,nv);
    const pt=[],nt=[],ut=[];
    for (const tid in tn) {
      if (!tn.hasOwnProperty(tid)) continue;
      const t2=tn[tid].text;
      if (pv[tid]&&!nv[tid]) pt.push(t2);
      else if (nv[tid]&&!pv[tid]) nt.push(t2);
      else ut.push(t2);
    }
    ut.forEach(t => looksNeg(t)?nt.push(t):pt.push(t));
    if (pt.length) row.positive=pt.join(" | ");
    if (nt.length) row.negative=nt.join(" | ");
  }
  if (!row.seed&&wf) {
    const nodes=wf.nodes||[];
    for (const nd of nodes) {
      const tp=nd.type||"", wv=nd.widgets_values||[];
      if ((tp==="KSampler"||tp==="KSamplerAdvanced")&&wv.length>=6) {
        if(!row.seed)row.seed=wv[0];if(!row.steps)row.steps=wv[3];if(!row.cfg)row.cfg=parseFloat(wv[4]).toFixed(1);if(!row.sampler)row.sampler=wv[1];if(!row.scheduler)row.scheduler=wv[2];if(!row.denoise)row.denoise=parseFloat(wv[5]).toFixed(2);
      }
      if ((tp==="CheckpointLoaderSimple"||tp==="CheckpointLoader")&&wv.length>=1&&!row.model) row.model=sn(wv[0]);
      if ((tp==="UNETLoader"||tp==="DiffusionModelLoader"||tp==="UnetLoaderGGUF")&&wv.length>=1&&!row.model) row.model=sn(wv[0]);
      if (isLora(tp)&&wv.length>=1){const ln4=sn(wv[0]);if(ln4)lm[ln4]={name:ln4,strength:wv.length>=2?parseFloat(wv[1]).toFixed(2):"1.00"};}
      if (isTE(tp)&&wv.length>=1){const t3=String(wv[0]).trim();if(t3&&t3.length>1){const title=(nd.title||"").toLowerCase();if(!row.positive&&!title.includes("neg"))row.positive=t3;else if(!row.negative)row.negative=t3;}}
    }
  }
  row.loras=Object.values(lm);
}

// ── Column definitions ───────────────────────────────────────────────────────
const DEFAULT_COLS = [
  { key:"preview",  label:"Preview",         vis:true,  numeric:false, w:90,  noSort:true, noExport:true },
  { key:"file",     label:"File",             vis:true,  numeric:false, w:150 },
  { key:"filesize", label:"Size (KB)",        vis:true,  numeric:true,  w:75  },
  { key:"filedate", label:"Date",             vis:true,  numeric:false, w:130 },
  { key:"seed",     label:"Seed",             vis:true,  numeric:true,  w:110 },
  { key:"steps",    label:"Steps",            vis:true,  numeric:true,  w:55  },
  { key:"cfg",      label:"CFG",              vis:true,  numeric:true,  w:50  },
  { key:"sampler",  label:"Sampler",          vis:true,  numeric:false, w:90  },
  { key:"scheduler",label:"Scheduler",        vis:true,  numeric:false, w:80  },
  { key:"size",     label:"Resolution",       vis:true,  numeric:false, w:90  },
  { key:"denoise",  label:"Denoise",          vis:false, numeric:true,  w:70  },
  { key:"model",    label:"Model",            vis:true,  numeric:false, w:130 },
  { key:"loras",    label:"LoRAs",            vis:true,  numeric:false, w:150 },
  { key:"positive", label:"Positive prompt",  vis:true,  numeric:false, w:200, prompt:true },
  { key:"negative", label:"Negative prompt",  vis:true,  numeric:false, w:160, prompt:true },
  { key:"status",   label:"Status",           vis:true,  numeric:false, w:60  },
];

function pad(n){ return n<10?"0"+n:String(n); }

function dl(name, content, type) {
  const b64 = btoa(unescape(encodeURIComponent(content)));
  const a = document.createElement("a");
  a.href = "data:"+type+";base64,"+b64;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ── Main component ───────────────────────────────────────────────────────────
export default function App() {
  const [rows, setRows] = useState([]);
  const [cols, setCols] = useState(DEFAULT_COLS.map(c=>({...c})));
  const [sortKey, setSortKey] = useState(null);
  const [sortDir, setSortDir] = useState(1);
  const [dragging, setDragging] = useState(false);
  const [tooltip, setTooltip] = useState({ text:"", x:0, y:0, show:false });
  const fileInputRef = useRef();
  const resizingRef = useRef(null);

  // ── File processing ────────────────────────────────────────────────────────
  const processFile = useCallback((file) => {
    if (!file.name.toLowerCase().endsWith(".png")) return;
    const row = { file: file.name, loras: [] };
    row.filesize = Math.round(file.size / 1024);
    if (file.lastModified) {
      const d = new Date(file.lastModified);
      row.filedate = d.getFullYear()+"-"+pad(d.getMonth()+1)+"-"+pad(d.getDate())+" "+pad(d.getHours())+":"+pad(d.getMinutes());
      row._filedateTs = file.lastModified;
    }
    const urlReader = new FileReader();
    urlReader.onload = e => {
      row.dataUrl = e.target.result;
      const img = new Image();
      img.onload = () => {
        row.imgW = img.naturalWidth;
        row.imgH = img.naturalHeight;
        row.size = row.imgW + "x" + row.imgH;
        const bufReader = new FileReader();
        bufReader.onload = e2 => {
          try {
            const chunks = parsePNG(e2.target.result);
            const data = extractComfy(chunks);
            if (data) { extractParams(row, data.workflow, data.prompt); row.status="ok"; }
            else row.status="no metadata";
          } catch(err) { row.status="error"; }
          setRows(prev => [...prev, row]);
        };
        bufReader.readAsArrayBuffer(file);
      };
      img.onerror = () => { row.imgW=1; row.imgH=1; row.size=""; row.status="error"; setRows(prev=>[...prev,row]); };
      img.src = e.target.result;
    };
    urlReader.readAsDataURL(file);
  }, []);

  const handleFiles = useCallback((files) => {
    Array.from(files).filter(f=>f.name.toLowerCase().endsWith(".png")).forEach(processFile);
  }, [processFile]);

  // ── Sorting ────────────────────────────────────────────────────────────────
  const getSortVal = (r, col) => {
    if (col.key==="loras") return r.loras ? r.loras.length : 0;
    if (col.key==="filedate") return r._filedateTs || 0;
    const v = r[col.key];
    if (v==null||v==="") return col.numeric ? -Infinity : "";
    if (col.numeric) return parseFloat(v)||0;
    return String(v).toLowerCase();
  };

  const sortedRows = (() => {
    if (!sortKey) return rows;
    const col = cols.find(c=>c.key===sortKey);
    if (!col) return rows;
    return [...rows].sort((a,b) => {
      const av=getSortVal(a,col), bv=getSortVal(b,col);
      if (av<bv) return -sortDir; if (av>bv) return sortDir; return 0;
    });
  })();

  const handleSort = (col) => {
    if (col.noSort) return;
    if (sortKey===col.key) setSortDir(d=>d*-1);
    else { setSortKey(col.key); setSortDir(1); }
  };

  // ── Column resizing ────────────────────────────────────────────────────────
  const startResize = (e, colKey) => {
    e.preventDefault(); e.stopPropagation();
    const startX = e.clientX;
    const startW = cols.find(c=>c.key===colKey).w;
    resizingRef.current = colKey;
    const onMove = ev => {
      const nw = Math.max(40, startW + (ev.clientX - startX));
      setCols(prev => prev.map(c => c.key===colKey ? {...c,w:nw} : c));
    };
    const onUp = () => { resizingRef.current=null; window.removeEventListener("mousemove",onMove); window.removeEventListener("mouseup",onUp); };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // ── Column width → table width ─────────────────────────────────────────────
  const totalW = cols.filter(c=>c.vis).reduce((s,c)=>s+c.w,0);

  // ── Toggle column ──────────────────────────────────────────────────────────
  const toggleCol = key => setCols(prev=>prev.map(c=>c.key===key?{...c,vis:!c.vis}:c));

  // ── Export ─────────────────────────────────────────────────────────────────
  const exportCSV = () => {
    const vc = cols.filter(c=>c.vis&&!c.noExport);
    const lines = [vc.map(c=>'"'+c.label+'"').join(",")];
    sortedRows.forEach(r => {
      const ls = (r.loras||[]).map(l=>l.name+":"+l.strength).join(" | ");
      lines.push(vc.map(c => { const v=c.key==="loras"?ls:(r[c.key]!=null?r[c.key]:""); return '"'+String(v).replace(/"/g,'""')+'"'; }).join(","));
    });
    dl("comfyui_prompts.csv", lines.join("\n"), "text/csv");
  };

  const exportJSON = () => {
    const vc = cols.filter(c=>c.vis&&!c.noExport);
    const out = sortedRows.map(r => {
      const o={};
      vc.forEach(c => {
        if (c.key==="loras") o[c.key]=(r.loras||[]).map(l=>l.name+":"+l.strength).join(" | ");
        else if (r[c.key]!=null) o[c.key]=r[c.key];
      });
      return o;
    });
    dl("comfyui_prompts.json", JSON.stringify(out,null,2), "application/json");
  };

  const clearAll = () => { setRows([]); setSortKey(null); setSortDir(1); if(fileInputRef.current) fileInputRef.current.value=""; };

  // ── Render preview cell ────────────────────────────────────────────────────
  const previewCol = cols.find(c=>c.key==="preview");
  const renderPreview = (r) => {
    const pw0 = Math.max(30, previewCol.w - 12);
    let pw=pw0, ph=pw0;
    if (r.imgW&&r.imgH) { const ar=r.imgW/r.imgH; if(ar>=1) ph=Math.round(pw/ar); else pw=Math.round(ph*ar); }
    return (
      <div style={{width:pw,height:ph,display:"flex",alignItems:"center",justifyContent:"center",overflow:"hidden",borderRadius:4,background:"#f0f0f0",flexShrink:0}}>
        {r.dataUrl ? <img src={r.dataUrl} width={pw} height={ph} style={{objectFit:"contain",borderRadius:3,display:"block"}} alt="" />
          : <span style={{fontSize:9,color:"#999"}}>-</span>}
      </div>
    );
  };

  // ── Cell renderer ──────────────────────────────────────────────────────────
  const renderCell = (col, r) => {
    if (col.key==="preview") return renderPreview(r);
    if (col.key==="loras") {
      if (!r.loras||!r.loras.length) return <span style={{color:"#999",fontSize:10}}>-</span>;
      return r.loras.map((l,i)=>(
        <span key={i} onMouseEnter={e=>setTooltip({text:l.name,x:e.clientX+12,y:e.clientY+12,show:true})} onMouseLeave={()=>setTooltip(t=>({...t,show:false}))}
          style={{display:"inline-block",padding:"1px 6px",borderRadius:4,fontSize:10,fontWeight:500,background:"#f0f0f0",color:"#555",border:"0.5px solid #ddd",margin:"1px 2px 1px 0",whiteSpace:"nowrap"}}>
          {l.name}<span style={{color:"#1a56db",fontSize:9,marginLeft:2}}>x{l.strength}</span>
        </span>
      ));
    }
    if (col.key==="status") {
      const ok=r.status==="ok";
      return <span style={{display:"inline-block",padding:"1px 6px",borderRadius:4,fontSize:10,fontWeight:500,background:ok?"#e8f0fe":"#fde8e8",color:ok?"#1a56db":"#c0392b"}}>{r.status}</span>;
    }
    const v = r[col.key]!=null ? String(r[col.key]) : "-";
    const needsTip = col.key==="file"||col.key==="model"||col.prompt;
    if (needsTip && v && v!=="-") {
      return (
        <span style={{display:"block",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:col.prompt?"normal":"nowrap",lineHeight:col.prompt?"1.4":undefined}}
          onMouseEnter={e=>setTooltip({text:v,x:e.clientX+12,y:e.clientY+12,show:true})}
          onMouseLeave={()=>setTooltip(t=>({...t,show:false}))}>
          {v}
        </span>
      );
    }
    return <span style={{display:"block",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{v}</span>;
  };

  const visCols = cols.filter(c=>c.vis);

  return (
    <div style={{fontFamily:"-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif",background:"#f5f4f0",minHeight:"100vh",padding:"1.25rem",color:"#1a1a1a",fontSize:13}}>
      {/* Title */}
      <h1 style={{fontSize:17,fontWeight:500,marginBottom:"1rem"}}>ComfyUI Metadata Inspector</h1>

      {/* Drop zone */}
      <div
        onClick={()=>fileInputRef.current?.click()}
        onDragOver={e=>{e.preventDefault();setDragging(true);}}
        onDragLeave={()=>setDragging(false)}
        onDrop={e=>{e.preventDefault();setDragging(false);handleFiles(e.dataTransfer.files);}}
        style={{border:`1.5px dashed ${dragging?"#888":"#bbb"}`,borderRadius:12,padding:"1.5rem 1rem",textAlign:"center",cursor:"pointer",background:dragging?"#efefef":"#fff",marginBottom:"0.75rem",transition:"background 0.15s"}}>
        <input ref={fileInputRef} type="file" multiple accept=".png" style={{display:"none"}} onChange={e=>handleFiles(e.target.files)} />
        <div style={{fontSize:14,fontWeight:500}}>Drop ComfyUI PNGs here</div>
        <div style={{fontSize:12,color:"#777",marginTop:4}}>or click to browse — multiple files supported</div>
      </div>

      {/* Toolbar */}
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:8,flexWrap:"wrap"}}>
        {[["Export CSV",exportCSV],["Export JSON",exportJSON],["Clear",clearAll]].map(([label,fn])=>(
          <button key={label} onClick={fn} disabled={rows.length===0}
            style={{fontSize:12,padding:"4px 12px",borderRadius:8,border:"0.5px solid #ccc",background:"transparent",cursor:rows.length===0?"default":"pointer",color:"#1a1a1a",opacity:rows.length===0?0.4:1}}>
            {label}
          </button>
        ))}
        <span style={{fontSize:12,color:"#777",marginLeft:"auto"}}>{rows.length>0&&`${rows.length} image${rows.length!==1?"s":""}`}</span>
      </div>

      {/* Column toggles */}
      <div style={{display:"flex",flexWrap:"wrap",gap:5,marginBottom:10,padding:"8px 10px",border:"0.5px solid #e0e0e0",borderRadius:8,background:"#f9f9f7",alignItems:"center"}}>
        <span style={{fontSize:11,color:"#777",marginRight:4}}>Columns:</span>
        {cols.map(col=>(
          <button key={col.key} onClick={()=>toggleCol(col.key)}
            style={{fontSize:11,padding:"2px 9px",borderRadius:4,border:`0.5px solid ${col.vis?"#ccc":"#ddd"}`,background:col.vis?"#fff":"transparent",cursor:"pointer",color:col.vis?"#1a1a1a":"#aaa",textDecoration:col.vis?"none":"line-through",opacity:col.vis?1:0.5}}>
            {col.label}
          </button>
        ))}
      </div>

      {/* Table */}
      {rows.length===0 ? (
        <div style={{textAlign:"center",padding:"1.5rem",color:"#777",fontSize:13}}>No images loaded yet</div>
      ) : (
        <div style={{overflowX:"auto",WebkitOverflowScrolling:"touch",border:"0.5px solid #e0e0e0",borderRadius:12,background:"#fff"}}>
          <table style={{borderCollapse:"collapse",fontSize:11.5,tableLayout:"fixed",width:totalW}}>
            <colgroup>
              {cols.map(col=>col.vis&&<col key={col.key} style={{width:col.w}} />)}
            </colgroup>
            <thead>
              <tr style={{background:"#f9f9f7"}}>
                {visCols.map(col=>(
                  <th key={col.key} style={{padding:0,textAlign:"left",fontWeight:500,fontSize:11,color:"#777",borderBottom:"0.5px solid #e0e0e0",position:"relative",whiteSpace:"nowrap",overflow:"hidden"}}>
                    <div onClick={()=>handleSort(col)} style={{display:"flex",alignItems:"center",padding:"6px 10px",cursor:col.noSort?"default":"pointer",userSelect:"none",gap:4}}
                      onMouseEnter={e=>{if(!col.noSort)e.currentTarget.style.color="#1a1a1a";}}
                      onMouseLeave={e=>{e.currentTarget.style.color="#777";}}>
                      <span>{col.label}</span>
                      {!col.noSort&&<span style={{fontSize:10,opacity:sortKey===col.key?1:0.35,color:sortKey===col.key?"#1a56db":"inherit"}}>
                        {sortKey===col.key?(sortDir===1?"^":"v"):"~"}
                      </span>}
                    </div>
                    <div onMouseDown={e=>startResize(e,col.key)}
                      style={{position:"absolute",right:0,top:0,bottom:0,width:5,cursor:"col-resize",zIndex:2,background:"transparent"}}
                      onMouseEnter={e=>e.currentTarget.style.background="#ccc"}
                      onMouseLeave={e=>e.currentTarget.style.background="transparent"} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((r,ri)=>(
                <tr key={ri} style={{}}>
                  {visCols.map(col=>(
                    <td key={col.key} style={{padding:col.key==="preview"?"3px 6px":"5px 10px",borderBottom:"0.5px solid #e0e0e0",verticalAlign:col.prompt?"top":"middle",overflow:"hidden",whiteSpace:col.prompt?"normal":"nowrap",textOverflow:"ellipsis",maxWidth:0,fontSize:col.prompt?11:11.5,lineHeight:col.prompt?"1.4":undefined}}>
                      {renderCell(col,r)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Tooltip */}
      {tooltip.show&&tooltip.text&&(
        <div style={{position:"fixed",left:tooltip.x,top:tooltip.y,background:"#fff",border:"0.5px solid #ccc",borderRadius:8,padding:"6px 10px",fontSize:11.5,color:"#1a1a1a",maxWidth:340,whiteSpace:"normal",lineHeight:1.5,pointerEvents:"none",zIndex:9999,wordBreak:"break-all",boxShadow:"0 2px 8px rgba(0,0,0,0.1)"}}>
          {tooltip.text}
        </div>
      )}
    </div>
  );
}
