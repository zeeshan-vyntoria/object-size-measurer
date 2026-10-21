import { useRef, useState } from 'react'
import './App.css'
import {
  IconAlert,
  IconBox,
  IconCheck,
  IconImage,
  IconRuler,
  IconShirt,
  IconSparkle,
  IconUpload,
} from './Icons'

const API_BASE = 'http://127.0.0.1:5050/api'

const MODES = [
  {
    id: 'clothing',
    label: 'Clothing',
    icon: IconShirt,
    pill: { icon: IconSparkle, text: 'Auto', cls: 'auto' },
    subtitle:
      "Upload a shirt photo — flat-laid or worn by a model, either works. A flat-laid photo with a known-size reference (e.g. a credit card) placed leftmost gets accurate numbers; without one, or for a worn photo, you'll get a rough size guess instead.",
  },
  {
    id: 'object',
    label: 'Generic object',
    icon: IconBox,
    pill: { icon: IconCheck, text: 'Accurate', cls: 'accurate' },
    subtitle:
      'Photograph any object top-down with a known-size reference placed leftmost in frame.',
  },
]

function App() {
  const [mode, setMode] = useState('clothing')
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [refWidth, setRefWidth] = useState(3.37)
  const [unit, setUnit] = useState('in')
  const [minArea, setMinArea] = useState(800)
  const [shoulderPct, setShoulderPct] = useState(16)
  const [chestPct, setChestPct] = useState(35)
  const [waistPct, setWaistPct] = useState(70)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const fileInputRef = useRef(null)

  const activeMode = MODES.find((m) => m.id === mode)

  function pickFile(f) {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    setPreviewUrl(URL.createObjectURL(f))
  }

  function switchMode(next) {
    setMode(next)
    setResult(null)
    setError(null)
  }

  async function handleSubmit() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const form = new FormData()
      form.append('image', file)
      form.append('min_area', minArea)
      form.append('ref_width', refWidth)
      form.append('unit', unit)
      let url = `${API_BASE}/measure`
      if (mode === 'clothing') {
        url = `${API_BASE}/measure-clothing`
        form.append('shoulder_pct', shoulderPct / 100)
        form.append('chest_pct', chestPct / 100)
        form.append('waist_pct', waistPct / 100)
      }

      const res = await fetch(url, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) {
        setError(data.error || 'Request failed')
      } else {
        setResult(data)
      }
    } catch (e) {
      setError('Could not reach the API — is server.py running on port 5050?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <IconRuler />
          </div>
          <div>
            <div className="brand-name">SizeSnap</div>
            <div className="brand-tag">Photo-based garment &amp; object measurement</div>
          </div>
        </div>
      </header>

      <main className="layout">
        <section className="panel panel-controls">
          <div className="mode-tabs">
            {MODES.map((m) => {
              const Icon = m.icon
              const PillIcon = m.pill.icon
              const isActive = mode === m.id
              return (
                <button
                  key={m.id}
                  className={'mode-tab' + (isActive ? ' active' : '')}
                  onClick={() => switchMode(m.id)}
                >
                  <Icon className="mode-tab-icon" />
                  <span className="mode-tab-label">{m.label}</span>
                  <span className={'mode-tab-pill ' + m.pill.cls}>
                    <PillIcon />
                    {m.pill.text}
                  </span>
                </button>
              )
            })}
          </div>

          <p className="mode-subtitle">{activeMode.subtitle}</p>

          {mode === 'clothing' && (
            <div className="callout callout-caution">
              <IconAlert className="callout-icon" />
              <div>
                <strong>Accuracy depends on the photo.</strong> A flat-laid
                shirt with a reference object alongside it gets real,
                calibrated measurements. A flat-laid shirt alone, or a photo
                of someone wearing the shirt, falls back to a rough size
                guess instead — there's no way to derive real-world size
                without something of known scale in frame, that's a hard
                limit, not a bug. Which case applies is detected
                automatically and shown clearly in the results below.
              </div>
            </div>
          )}

          <div
            className={'dropzone' + (dragging ? ' drag' : '') + (file ? ' has-file' : '')}
            onClick={() => fileInputRef.current.click()}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              pickFile(e.dataTransfer.files[0])
            }}
          >
            {previewUrl ? (
              <img className="dropzone-preview" src={previewUrl} alt="preview" />
            ) : (
              <>
                <IconUpload className="dropzone-icon" />
                <div className="dropzone-title">Click or drop a photo here</div>
                <div className="dropzone-hint">JPG or PNG</div>
              </>
            )}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => pickFile(e.target.files[0])}
            />
          </div>
          {file && <div className="filename">{file.name}</div>}

          <div className="row">
            <div className="field">
              <label>Reference width{mode === 'clothing' ? ' (if present)' : ''}</label>
              <input
                type="number"
                step="0.01"
                value={refWidth}
                onChange={(e) => setRefWidth(e.target.value)}
              />
            </div>
            <div className="field">
              <label>Unit</label>
              <input type="text" value={unit} onChange={(e) => setUnit(e.target.value)} />
            </div>
          </div>

          {mode === 'clothing' && (
            <div className="row">
              <div className="field">
                <label>Shoulder level (% down from top)</label>
                <input
                  type="number"
                  step="1"
                  value={shoulderPct}
                  onChange={(e) => setShoulderPct(e.target.value)}
                />
                <span className="field-hint">
                  Raise this if the shirt has a tall/popped collar — the shoulder line
                  sits lower than the collar tip.
                </span>
              </div>
            </div>
          )}

          <details className="advanced">
            <summary>Advanced settings</summary>
            <div className="row">
              <div className="field">
                <label>Min area (px²)</label>
                <input
                  type="number"
                  step="50"
                  value={minArea}
                  onChange={(e) => setMinArea(e.target.value)}
                />
              </div>
              {mode === 'clothing' && (
                <>
                  <div className="field">
                    <label>Chest level (%)</label>
                    <input
                      type="number"
                      step="1"
                      value={chestPct}
                      onChange={(e) => setChestPct(e.target.value)}
                    />
                  </div>
                  <div className="field">
                    <label>Waist level (%)</label>
                    <input
                      type="number"
                      step="1"
                      value={waistPct}
                      onChange={(e) => setWaistPct(e.target.value)}
                    />
                  </div>
                </>
              )}
            </div>
          </details>

          <button className="primary" onClick={handleSubmit} disabled={!file || loading}>
            {loading ? (
              <>
                <span className="spinner" /> Working…
              </>
            ) : (
              'Run scan'
            )}
          </button>

          {error && (
            <div className="callout callout-error">
              <IconAlert className="callout-icon" />
              <div>{error}</div>
            </div>
          )}
        </section>

        <section className="panel panel-results">
          {!result && !error && (
            <div className="empty-state">
              <IconImage className="empty-icon" />
              <div className="empty-title">Results will appear here</div>
              <div className="empty-hint">
                Choose a mode, upload a photo, and run a scan to see measurements.
              </div>
            </div>
          )}
          {result && mode === 'clothing' && <ClothingResults result={result} />}
          {result && mode === 'object' && <ObjectResults result={result} />}
        </section>
      </main>
    </div>
  )
}

function ResultHeader({ icon: Icon, title, tone }) {
  return (
    <div className={'result-header ' + (tone || '')}>
      <Icon />
      <span>{title}</span>
    </div>
  )
}

function ClothingResults({ result }) {
  if (result.accuracy === 'measured') {
    return <MeasuredClothing result={result} />
  }
  return <EstimatedClothing result={result} />
}

function MeasuredClothing({ result }) {
  return (
    <>
      <ResultHeader
        icon={IconCheck}
        title="Measured — reference object detected"
        tone="accurate"
      />
      <div className="stats">
        <div className="stat">
          <div className="value">{result.length}</div>
          <div className="label">Shirt length ({result.unit})</div>
        </div>
        <div className="stat">
          <div className="value">{result.shoulder_width}</div>
          <div className="label">Shoulder width ({result.unit})</div>
        </div>
        <div className="stat">
          <div className="value">{result.sleeve_length !== null ? result.sleeve_length : '—'}</div>
          <div className="label">Sleeve length, approx ({result.unit})</div>
        </div>
        <div className="stat">
          <div className="value">
            {result.sleeve_with_shoulder_length !== null ? result.sleeve_with_shoulder_length : '—'}
          </div>
          <div className="label">Sleeve with shoulder, approx ({result.unit})</div>
        </div>
      </div>

      <div className="section-title">Circumferences (flat measurement × 2)</div>
      <table>
        <thead>
          <tr>
            <th>Area</th>
            <th>Flat ({result.unit})</th>
            <th>Circumference ({result.unit})</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Chest{result.armpit_detected ? '' : ' (estimated level)'}</td>
            <td>{result.chest_width_flat}</td>
            <td>{result.chest_circumference}</td>
          </tr>
          <tr>
            <td>Waist</td>
            <td>{result.waist_width_flat}</td>
            <td>{result.waist_circumference}</td>
          </tr>
          <tr>
            <td>Bottom width (hem)</td>
            <td>{result.bottom_width_flat}</td>
            <td>{result.bottom_width_circumference}</td>
          </tr>
          <tr>
            <td>Neck{result.neck_width_flat !== null ? ' (rough estimate)' : ''}</td>
            <td>{result.neck_width_flat !== null ? result.neck_width_flat : 'not detected'}</td>
            <td>{result.neck_circumference_est !== null ? result.neck_circumference_est : '—'}</td>
          </tr>
        </tbody>
      </table>
      <div className="callout callout-note">
        Wrist circumference isn't measured yet — it needs reliable cuff
        detection for long sleeves, which this contour-based approach can't
        do accurately.
      </div>

      {result.annotated_image && (
        <>
          <div className="section-title">Annotated image</div>
          <img className="preview" src={result.annotated_image} alt="annotated" />
        </>
      )}
    </>
  )
}

function EstimatedClothing({ result }) {
  const isWorn = result.photo_type === 'worn'
  const rows = [
    ['Length', result.estimated_length_in],
    ['Shoulder width', result.estimated_shoulder_width_in],
    ['Sleeve length', result.estimated_sleeve_length_in],
    ['Neck circumference', result.estimated_neck_circumference_in],
  ]
  return (
    <>
      <ResultHeader
        icon={IconAlert}
        title={isWorn ? 'Estimated — worn-by-model photo' : 'Estimated — no reference object detected'}
        tone="rough"
      />
      <div className="stats">
        <div className="stat stat-big">
          <div className="value">{result.estimated_size_label}</div>
          <div className="label">Closest standard size guess</div>
        </div>
      </div>

      <div className="section-title">From a real published tee size chart</div>
      <table>
        <thead>
          <tr>
            <th>Area</th>
            <th>Flat (in)</th>
            <th>Circumference (in)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Chest</td>
            <td>~{result.estimated_chest_flat_in}</td>
            <td>~{result.estimated_chest_circumference_in}</td>
          </tr>
          <tr>
            <td>Bottom width (hem)</td>
            <td>~{result.estimated_bottom_width_flat_in}</td>
            <td>~{result.estimated_bottom_width_circumference_in}</td>
          </tr>
        </tbody>
      </table>
      <table>
        <thead>
          <tr>
            <th>Other typical values for this size</th>
            <th>Typical (in)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <td>{label}</td>
              <td>~{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="callout callout-caution">
        <IconAlert className="callout-icon" />
        <div>
          <strong>These are not this garment's measurements.</strong>{' '}
          {isWorn ? (
            <>
              This photo shows someone <em>wearing</em> the shirt, so there's no
              flat-laid garment shape to measure and no reference object is
              practical either. This instead reads the wearer's body
              proportions (shoulder width and arm length, from detected body
              landmarks) and matches them to a real published tee size chart —
              it's reading the person's build, not the garment's actual cut, so
              a loose or fitted style will read the same way.
            </>
          ) : (
            <>
              Chest, length, and bottom width come from a real published tee
              size chart for whichever size the silhouette happened to match
              best — shoulder and neck aren't on that chart, so those two are
              rougher placeholders.
            </>
          )}{' '}
          Either way it's a fallback suggestion, not something to show a
          customer as fact.{' '}
          {isWorn
            ? 'For real accuracy, photograph the shirt laid flat instead, with a reference object in frame.'
            : 'Add a reference object to the photo for real accuracy.'}
        </div>
      </div>
      {result.annotated_image && (
        <>
          <div className="section-title">{isWorn ? 'Detected pose' : 'Detected outline'}</div>
          <img className="preview" src={result.annotated_image} alt="annotated" />
        </>
      )}
    </>
  )
}

function ObjectResults({ result }) {
  return (
    <>
      <ResultHeader icon={IconCheck} title="Measurement results" tone="accurate" />
      <table>
        <thead>
          <tr>
            <th>Label</th>
            <th>Width ({result.unit})</th>
            <th>Height ({result.unit})</th>
          </tr>
        </thead>
        <tbody>
          {result.results.map((r, i) => (
            <tr key={i}>
              <td>{r.label}</td>
              <td>{r.width}</td>
              <td>{r.height}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {result.annotated_image && (
        <>
          <div className="section-title">Annotated image</div>
          <img className="preview" src={result.annotated_image} alt="annotated" />
        </>
      )}
    </>
  )
}

export default App
