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

// Live version of analyze(): streams a fresh eval snapshot after every
// depth Stockfish reports (depth counting up, eval/best-moves refining in
// real time) instead of waiting for one final result.
//
// onUpdate(result) is called once per depth update; result has the same
// shape as analyze()'s resolved value, plus `depth` and `done`.
// unlimited: if true, the search ignores its normal time cap and runs to
// full depth however long that takes -- the caller is responsible for
// closing the connection (e.g. via the returned close function) when it
// no longer needs the result, since there's no time limit to end it.
// Returns a function you can call to close the connection early (e.g. if
// the user navigates away or a newer analysis supersedes this one).
export function analyzeStream(fen, turn, numMoves = 3, onUpdate, onError, unlimited = false) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const ws = new WebSocket(`${protocol}//${window.location.host}${BASE}/ws/analyze`)

  ws.onopen = () => {
    ws.send(JSON.stringify({ fen, turn, num_moves: numMoves, unlimited }))
  }

  ws.onmessage = (event) => {
    let msg
    try {
      msg = JSON.parse(event.data)
    } catch {
      return
    }
    if (msg.success === false) {
      onError?.(new Error(msg.error || 'Analysis failed'))
      ws.close()
      return
    }
    onUpdate?.(msg)
    if (msg.done) ws.close()
  }

  ws.onerror = () => {
    onError?.(new Error('Analysis connection failed'))
  }

  return () => ws.close()
}
