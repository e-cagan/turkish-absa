// Human-readable Turkish labels for the closed aspect taxonomy.
export const ASPECT_LABEL = {
  urun_kalitesi: 'Ürün kalitesi',
  fiyat: 'Fiyat',
  kargo_teslimat: 'Kargo & teslimat',
  ambalaj: 'Ambalaj',
  koku: 'Koku',
  kalicilik: 'Kalıcılık',
  musteri_hizmetleri: 'Müşteri hizmetleri',
}

// Sentiment -> display label + colors (CSS variables defined in styles.css).
export const SENT = {
  pozitif: { label: 'pozitif', color: 'var(--pos)', bg: 'var(--pos-bg)' },
  negatif: { label: 'negatif', color: 'var(--neg)', bg: 'var(--neg-bg)' },
  notr: { label: 'nötr', color: 'var(--neu)', bg: 'var(--neu-bg)' },
}

export const EXAMPLES = [
  'Kargo çok hızlıydı ama ürün berbat, kokusu da ağır geldi.',
  'Fiyatına göre gayet iyi, cildimi yumuşattı. Paketleme de özenliydi.',
  'İade etmek istedim ama satıcı hiç dönüş yapmadı, rezalet bir deneyim.',
]
