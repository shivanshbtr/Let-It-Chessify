const BASE = '/api'

async function post(path, body, isFormData = false) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    body: isFormData ? body : JSON.stringify(body),
    headers: isFormData ? {} : { 'Content-Type': 'application/json' },
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || 'Request failed')
  }
  return res.json()
}

export async function detectCorners(file, isPhysical) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('is_physical', String(isPhysical))
  return post('/detect-corners', fd, true)
}

export async function refineCorners(imageB64, corners, isPhysical) {
  return post('/refine-corners', { image_b64: imageB64, corners, is_physical: isPhysical })
}

export async function classify(imageB64, grid, isPhysical) {
  return post('/classify', { image_b64: imageB64, grid, is_physical: isPhysical })
}

export async function editFen(squareLabels, turn) {
  return post('/edit-fen', { square_labels: squareLabels, turn })
}

export async function analyze(fen, turn, numMoves = 3) {
  return post('/analyze', { fen, turn, num_moves: numMoves })
}
