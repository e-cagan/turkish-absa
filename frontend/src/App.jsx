import { useState } from 'react'
import Hero from './components/Hero.jsx'
import ReviewInput from './components/ReviewInput.jsx'
import Results from './components/Results.jsx'
import { analyzeReview } from './api.js'

export default function App() {
  const [text, setText] = useState('')
  const [aspects, setAspects] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleAnalyze() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await analyzeReview(text)
      setAspects([...data.aspects].sort((a, b) => b.confidence - a.confidence))
    } catch {
      setError('Servise ulaşılamadı. serve.py çalışıyor mu?')
      setAspects(null)
    } finally {
      setLoading(false)
    }
  }

  function pickExample(ex) {
    setText(ex)
    setAspects(null)
    setError(null)
  }

  return (
    <div className="wrap">
      <Hero />
      <ReviewInput
        text={text}
        setText={setText}
        onAnalyze={handleAnalyze}
        loading={loading}
        onPickExample={pickExample}
      />
      <Results aspects={aspects} error={error} />
      <footer className="footer">
        BERTurk · aspect-based sentiment · <span>turkish-absa</span>
      </footer>
    </div>
  )
}
