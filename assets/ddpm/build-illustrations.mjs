// Original educational vector illustrations. No model inference or experimental data.
// The noise tiles are stylized, deterministic dots, not a simulated diffusion trajectory.
import { mkdir, writeFile } from 'node:fs/promises';
const out = new URL('./', import.meta.url);
await mkdir(out, {recursive:true});
const C={bg:'#F6F3EB',ink:'#253D3A',muted:'#566760',green:'#147D72',orange:'#B75B38',line:'#D5DCD3'};
const text=(x,y,s,size=24,color=C.ink,extra='')=>`<text x="${x}" y="${y}" font-size="${size}" fill="${color}" ${extra}>${s}</text>`;
const box=(x,y,w,h,fill,rx=18)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}"/>`;
const arrow=(x,y,w,color=C.green)=>`<path d="M${x} ${y}h${w}" stroke="${color}" stroke-width="3" fill="none"/><path d="m${x+w-8} ${y-6} 8 6-8 6" stroke="${color}" stroke-width="3" fill="none"/>`;
const svg=(w,h,title,body)=>`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-labelledby="title"><title id="title">${title}</title><style>text{font-family:'Microsoft YaHei','Noto Sans CJK SC','PingFang SC',sans-serif}</style>${box(0,0,w,h,C.bg,24)}${body}</svg>`;
function house(x,y,s=1,opacity=1){return `<g transform="translate(${x} ${y}) scale(${s})" opacity="${opacity}">${box(0,0,170,150,'#DAE8DE',12)}<circle cx="137" cy="29" r="13" fill="#E1B65D"/><path d="M0 118Q50 78 91 110T170 108V150H0Z" fill="#A6C6AA"/><path d="M47 73H126V135H47Z" fill="#F8E5C6"/><path d="M33 77 86 29 140 77Z" fill="#B75B38"/><path d="M76 98H97V135H76Z" fill="#43766C"/><path d="M53 89H67V105H53ZM106 89H120V105H106Z" fill="#7EAFB7"/></g>`;}
function dots(x,y,w,h,seed,opacity=1){let s=seed; const rand=()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296;};let r='';for(let j=0;j<h;j+=5)for(let i=0;i<w;i+=5){const v=Math.floor(40+rand()*190);r+=`<rect x="${x+i}" y="${y+j}" width="5" height="5" fill="rgb(${v},${v},${v})" opacity="${opacity}"/>`;}return r;}
let b=text(40,45,'视觉生成入门 / 01',17,C.green,'letter-spacing="2"')+text(40,100,'DDPM：从一张小房子开始',36,C.ink,'font-weight="700"')+text(40,139,'先把图弄乱，再学着一步步画清楚。',21,C.muted);
const xs=[40,280,520,760], labels=['看得清','有些杂点','快认不出了','几乎只剩噪声'];
xs.forEach((x,i)=>{b+=house(x,180,1,i===3?0:1);if(i)b+=dots(x,180,170,150,23+i,[0,.25,.66,1][i]);b+=text(x,365,labels[i],21);if(i<3)b+=arrow(x+187,255,35);});
b+=text(40,412,'加噪：由我们规定，不需要神经网络。',20,C.green)+text(40,447,'图像会变得怎样？先用这幅示意图建立直觉。',17,C.muted)+text(40,480,'原创概念插画；杂点不是实际高斯加噪结果。',15,C.muted);
await writeFile(new URL('01-cover.svg',out),svg(980,510,'小房子逐渐被杂点遮住：原图、少量噪声、大量噪声、近似纯噪声。概念示意，非实验结果。',b));
b=text(36,48,'训练时：我们既出题，也知道答案',30,C.ink,'font-weight="700"')+text(36,82,'每次换一张图、换一份噪声，再换一个难度。',20,C.muted);
b+=house(38,125,.66)+text(38,247,'干净图',20)+text(172,188,'+',38,C.orange)+dots(213,125,110,100,42)+text(213,247,'已知噪声',20);
b+=arrow(345,174,55)+box(428,123,240,108,'#DFEBE5')+text(451,164,'混合：得到带噪图',21)+text(451,199,'并告诉网络噪声等级',18,C.muted);
b+=arrow(690,174,55)+box(770,123,170,108,C.green)+text(800,169,'去噪网络',24,'#FFFFFF')+text(802,204,'猜噪声',18,'#FFFFFF');
b+=`<path d="M855 233V298H548" stroke="${C.green}" stroke-width="3" fill="none"/>`+box(427,278,242,102,'#F2DFD1')+text(451,320,'预测与答案比一比',21)+text(451,351,'差多少，就据此调整网络',17,C.muted);
b+=`<path d="M268 254V330H406" stroke="${C.orange}" stroke-width="3" stroke-dasharray="7 6" fill="none"/>`+arrow(396,330,22,C.orange);
b+=text(39,421,'为什么有答案？因为噪声是我们自己加的。',22,C.ink)+text(39,459,'网络看到的是带噪图和时间步；干净图、真实噪声不作为网络输入。',17,C.muted);
await writeFile(new URL('02-training.svg',out),svg(980,490,'训练流程：干净图与已知噪声混合，网络只看带噪图和噪声等级，预测噪声，再与已知答案比较并更新参数。',b));
b=text(36,48,'生成时：没有原图，也没有标准答案',30,C.ink,'font-weight="700"')+text(36,82,'先抽一张随机噪声图，再反复调用同一个已训练网络。',20,C.muted);
b+=dots(40,132,150,135,73)+text(40,306,'随机起点',21)+arrow(208,202,50);
b+=box(280,132,386,135,'#DFEBE5')+text(308,173,'预测噪声 → 算出下一步',24)+text(308,211,'向更清晰的图像前进一步',21)+text(308,245,'中间步骤仍会加入少量随机性',17,C.muted);
b+=`<path d="M595 272V337H350V282" stroke="${C.green}" stroke-width="3" fill="none"/><path d="m344 290 6-8 6 8" stroke="${C.green}" stroke-width="3" fill="none"/>`+text(421,366,'重复许多次',20,C.green)+arrow(689,202,50)+house(762,132,.9)+text(762,306,'一张新图',21);
b+=text(39,415,'不是找回某一张丢失的图片，而是生成新的样本。',22)+text(39,454,'右侧房子仅用于解释流程，并非模型生成结果；训练图也不需要在生成时提供。',16,C.muted);
await writeFile(new URL('03-sampling.svg',out),svg(980,485,'生成流程：随机噪声进入已训练网络，反复预测和更新，最后得到新图像。房子仅为示意，并非模型生成结果。',b));
console.log('Created 3 SVG illustrations.');
