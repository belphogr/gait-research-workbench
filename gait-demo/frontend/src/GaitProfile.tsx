import {Activity, CircleHelp, Footprints, Gauge, MoveHorizontal, ScanLine, ShieldCheck} from 'lucide-react';
import {PolarAngleAxis, PolarGrid, Radar, RadarChart, ResponsiveContainer, Tooltip} from 'recharts';

export type ProfileItem = {key:string;label:string;score:number|null;value:string;formula:string;source:string};
export type Profile = {version:string;items:ProfileItem[];note:string};

const icons = {quality:ScanLine,rhythm:Activity,symmetry:MoveHorizontal,stability:Gauge,mobility:Footprints};

export default function GaitProfile({profile}:{profile:Profile}) {
  const complete = profile.items.every(item => item.score !== null);
  return <section className="profile-section" aria-label="步态观察画像">
    <div className="profile-heading">
      <div className="profile-title"><span className="profile-title-icon"><ShieldCheck size={22}/></span><div><h3>步态观察画像</h3></div></div>
      <span className="profile-tag">科研演示</span>
    </div>
    <div className="profile-layout">
      <div className="profile-radar">
        {complete ? <ResponsiveContainer width="100%" height={330}>
          <RadarChart data={profile.items} outerRadius="69%" margin={{top:12,right:20,bottom:12,left:20}}>
            <PolarGrid stroke="#dbe8ed" strokeDasharray="4 4"/>
            <PolarAngleAxis dataKey="label" tick={{fill:'#4b6574',fontSize:12,fontWeight:600}}/>
            <Radar dataKey="score" stroke="#1599b4" fill="#46c2d1" fillOpacity={0.28} strokeWidth={2.8} dot={{r:4,fill:'#fff',stroke:'#1599b4',strokeWidth:2}} isAnimationActive={false}/>
            <Tooltip formatter={(value) => [`${Number(value).toFixed(1)} / 100`, '展示分']} contentStyle={{border:'1px solid #dbe9ee',borderRadius:12,boxShadow:'0 12px 32px #153d5016'}}/>
          </RadarChart>
        </ResponsiveContainer> : <div className="profile-unavailable">部分维度缺少有效数据，暂不绘制完整画像</div>}
        <span className="profile-axis-note">0–100 为展示刻度 · 非临床评分</span>
      </div>
      <div className="profile-metrics">
        {profile.items.map(item => {const Icon=icons[item.key as keyof typeof icons] || Activity;return <div className="profile-metric" key={item.key} title={item.formula}>
          <span className="profile-metric-icon"><Icon size={19}/></span>
          <div className="profile-metric-copy"><span>{item.label}</span><small>{item.value}</small></div>
          <strong>{item.score === null ? '—' : Math.round(item.score)}<small>{item.score === null ? '' : '/100'}</small></strong>
        </div>})}
      </div>
    </div>
    <details className="profile-method"><summary><CircleHelp size={15}/> 计算依据</summary><p>{profile.note}</p><div>{profile.items.map(item=><p key={item.key}><b>{item.label}</b>：{item.formula} · {item.source}</p>)}</div></details>
  </section>;
}
