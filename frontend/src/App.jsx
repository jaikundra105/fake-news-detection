import { useEffect, useState } from 'react'
import { ShieldCheck, Newspaper, Search, RotateCcw, Activity, BrainCircuit, AlertTriangle } from 'lucide-react'

const API = 'http://localhost:8000'
const examples = [
  "Scientists at a university published a peer-reviewed study describing a new method for improving battery efficiency. The researchers said further testing is needed before the technology can be used commercially.",
  "BREAKING: A secret miracle discovery will make everyone rich overnight. Share this shocking story immediately before the government deletes it!"
]

export default function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [apiReady, setApiReady] = useState(false)

  useEffect(() => {
    fetch(`${API}/api/health`).then(r => r.json()).then(d => setApiReady(d.models_ready)).catch(() => setApiReady(false))
  }, [])

  async function analyze() {
    setError(''); setResult(null)
    if (text.trim().length < 20) { setError('Please enter at least 20 characters of news content.'); return }
    setLoading(true)
    try {
      const response = await fetch(`${API}/api/predict`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({text})
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Prediction failed')
      setResult(data)
    } catch (e) { setError(e.message || 'Could not connect to the backend.') }
    finally { setLoading(false) }
  }

  function reset() { setText(''); setResult(null); setError('') }
  const isReal = result?.prediction === 'REAL'

  return <div className="page">
    <header className="topbar">
      <a className="brand" href="#"><span className="brand-icon"><ShieldCheck size={22}/></span> TruthLens<span className="brand-dot">.</span></a>
      <div className="status"><span className={apiReady ? 'dot online' : 'dot'}></span>{apiReady ? 'Models online' : 'Backend / models offline'}</div>
    </header>

    <main>
      <section className="hero">
        <div className="eyebrow"><Activity size={14}/> NLP CLASSIFICATION PROJECT</div>
        <h1>Separate facts from <span>fiction.</span></h1>
        <p className="sub">Analyze news content using two machine learning models and compare their predictions in seconds.</p>
      </section>

      <section className="workspace">
        <div className="input-card">
          <div className="card-heading">
            <div><div className="icon-box"><Newspaper size={19}/></div></div>
            <div><h2>Analyze a news article</h2><p>Paste a headline or the full article below.</p></div>
          </div>
          <textarea value={text} onChange={e => setText(e.target.value)} placeholder="Paste news content here..." maxLength={30000}/>
          <div className="textarea-footer"><span>{text.length.toLocaleString()} / 30,000 characters</span><button className="text-button" onClick={reset}><RotateCcw size={14}/> Clear</button></div>
          <div className="examples"><span>Try an example:</span><button onClick={() => setText(examples[0])}>News-style sample</button><button onClick={() => setText(examples[1])}>Sensational sample</button></div>
          <button className="analyze" onClick={analyze} disabled={loading}><Search size={17}/>{loading ? 'Analyzing...' : 'Analyze news'}</button>
          {error && <div className="error"><AlertTriangle size={16}/>{error}</div>}
        </div>

        <div className="result-card">
          <div className="result-title"><div><div className="icon-box"><BrainCircuit size={19}/></div></div><div><h2>Analysis results</h2><p>Model predictions and confidence</p></div></div>
          {!result ? <div className="empty"><div className="empty-icon"><Activity size={27}/></div><h3>Waiting for analysis</h3><p>Your classification results will appear here after you analyze an article.</p></div> :
            <>
              <div className={`verdict ${isReal ? 'real' : 'fake'}`}>
                <span className="verdict-label">ENSEMBLE VERDICT</span><strong>{result.prediction}</strong>
                <span>{result.agreement ? 'Both models agree' : 'Models disagree — review carefully'}</span>
              </div>
              <div className="model-results">
                {Object.entries(result.models).map(([key, value]) => <div className="model-row" key={key}>
                  <div className="model-meta"><strong>{key === 'naive_bayes' ? 'Multinomial Naive Bayes' : 'Logistic Regression'}</strong><span className={value.label === 'REAL' ? 'tag real-tag' : 'tag fake-tag'}>{value.label}</span></div>
                  <div className="bar"><span style={{width: `${value.confidence}%`}}/></div>
                  <div className="confidence">{value.confidence}% confidence for predicted class</div>
                </div>)}
              </div>
              <p className="disclaimer">{result.disclaimer}</p>
            </>
          }
        </div>
      </section>

      <section className="how">
        <h2>What happens behind the scenes?</h2>
        <div className="steps">
          <div><span>01</span><strong>Preprocessing</strong><p>Clean text, lowercase, remove noise and stopwords.</p></div>
          <div><span>02</span><strong>TF-IDF</strong><p>Convert words and phrases into numerical features.</p></div>
          <div><span>03</span><strong>Classification</strong><p>Two trained models predict the news label independently.</p></div>
          <div><span>04</span><strong>Comparison</strong><p>Display both predictions and whether they agree.</p></div>
        </div>
      </section>
      <footer>TRUTHLENS · NLP MINI PROJECT <span>Educational demonstration — not a professional fact-checking service.</span></footer>
    </main>
  </div>
}
