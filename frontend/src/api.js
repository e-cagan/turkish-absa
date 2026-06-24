// API base is empty by default: dev uses the Vite proxy, prod is same-origin.
// Override with VITE_API_BASE if the API lives on a different host.
const BASE = import.meta.env.VITE_API_BASE ?? ''

export async function analyzeReview(text) {
  const res = await fetch(`${BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}
