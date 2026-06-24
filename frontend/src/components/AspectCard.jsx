import { ASPECT_LABEL, SENT } from '../constants.js'

export default function AspectCard({ aspect, index }) {
  const s = SENT[aspect.sentiment] || SENT.notr
  const pct = Math.round(aspect.confidence * 100)
  return (
    <div className="aspect" style={{ animationDelay: `${index * 40}ms` }}>
      <div className="aspect-top">
        <span className="aspect-name">{ASPECT_LABEL[aspect.aspect] || aspect.aspect}</span>
        <span className="pill" style={{ color: s.color, background: s.bg }}>
          {s.label}
        </span>
      </div>
      <div className="meter">
        <span style={{ width: pct + '%', background: s.color }} />
      </div>
      <div className="conf">güven {pct}%</div>
    </div>
  )
}
