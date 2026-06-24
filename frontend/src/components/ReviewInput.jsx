import { EXAMPLES } from '../constants.js'

export default function ReviewInput({ text, setText, onAnalyze, loading, onPickExample }) {
  return (
    <section className="card">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Örn. Kargo hızlıydı ama ürünün kokusu çok ağır geldi…"
        aria-label="Ürün yorumu"
      />
      <div className="row">
        <button className="analyze" onClick={onAnalyze} disabled={loading || !text.trim()}>
          {loading ? 'Analiz ediliyor…' : 'Analiz et'}
        </button>
        <div className="examples">
          {EXAMPLES.map((ex, i) => (
            <button key={i} className="chip" onClick={() => onPickExample(ex)}>
              {ex.length > 38 ? ex.slice(0, 38) + '…' : ex}
            </button>
          ))}
        </div>
      </div>
    </section>
  )
}
