import AspectCard from './AspectCard.jsx'
import { SENT } from '../constants.js'

export default function Results({ aspects, error }) {
  if (error) {
    return (
      <section className="results">
        <div className="error">{error}</div>
      </section>
    )
  }
  if (!aspects) return null

  const tally = aspects.reduce((m, a) => ({ ...m, [a.sentiment]: (m[a.sentiment] || 0) + 1 }), {})
  const summary = ['pozitif', 'notr', 'negatif']
    .filter((k) => tally[k])
    .map((k) => `${tally[k]} ${SENT[k].label}`)
    .join(' · ')

  return (
    <section className="results">
      <div className="results-head">
        <h2>Bulunan yönler</h2>
        <span className="count">{aspects.length ? summary : '0 yön'}</span>
      </div>
      {aspects.length === 0 ? (
        <div className="empty">Bu yorumda belirgin bir yön bulunamadı.</div>
      ) : (
        aspects.map((a, i) => <AspectCard key={i} aspect={a} index={i} />)
      )}
    </section>
  )
}
