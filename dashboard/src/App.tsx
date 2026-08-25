import { useEffect, useMemo, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8010'

type RazorpayCheckout = {
  open: () => void
  on: (event: string, handler: (response: unknown) => void) => void
}

declare global {
  interface Window {
    Razorpay?: new (options: Record<string, unknown>) => RazorpayCheckout
  }
}

type AuditEvent = {
  timestamp: string
  event: string
  detail: string
  tone: 'neutral' | 'positive' | 'warning'
}

type RecoveryCase = {
  id: string
  customer_name: string
  amount: number
  language: string
  payment_method: string
  failure_reason: string
  previous_payments: number
  successful_payments: number
  customer_ltv: number
  recovery_probability: number
  strategy: string
  status: 'recovered' | 'active' | 'stopped' | 'escalated'
  interventions: number
  recovered_amount: number
  audit: AuditEvent[]
}

type Dashboard = {
  transactions: number
  failed_transactions: number
  revenue_at_risk: number
  revenue_recovered: number
  net_recovered: number
  recovery_rate: number
  intervention_cost: number
  interventions: number
  recovered_cases: number
  stopped_cases: number
  escalated_cases: number
  policy_violations: number
  duplicate_actions: number
}

type RazorpayStatus = {
  configured: boolean
  webhook_configured: boolean
  mode: 'test' | null
  key_id: string | null
}

const money = (value: number) =>
  new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: value % 1 === 0 ? 0 : 2,
  }).format(value)

const label = (value: string) =>
  value.replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase())

function App() {
  const [metrics, setMetrics] = useState<Dashboard | null>(null)
  const [cases, setCases] = useState<RecoveryCase[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [razorpayStatus, setRazorpayStatus] = useState<RazorpayStatus | null>(null)
  const [paymentLoading, setPaymentLoading] = useState(false)
  const [paymentMessage, setPaymentMessage] = useState('')

  const selected = useMemo(
    () => cases.find((item) => item.id === selectedId) ?? cases[0],
    [cases, selectedId],
  )

  async function loadDashboard() {
    try {
      setError('')
      const [metricsResponse, casesResponse, razorpayResponse] = await Promise.all([
        fetch(`${API_URL}/api/dashboard`),
        fetch(`${API_URL}/api/cases?limit=25`),
        fetch(`${API_URL}/api/razorpay/status`),
      ])
      if (!metricsResponse.ok || !casesResponse.ok || !razorpayResponse.ok) {
        throw new Error('Backend unavailable')
      }
      const nextMetrics = (await metricsResponse.json()) as Dashboard
      const nextCases = (await casesResponse.json()) as RecoveryCase[]
      const nextRazorpayStatus = (await razorpayResponse.json()) as RazorpayStatus
      setMetrics(nextMetrics)
      setCases(nextCases)
      setRazorpayStatus(nextRazorpayStatus)
      setSelectedId((current) => current ?? nextCases[0]?.id ?? null)
    } catch {
      setError('Start the FastAPI backend on port 8010, then refresh this page.')
    } finally {
      setLoading(false)
    }
  }

  async function runSimulation() {
    setRunning(true)
    setError('')
    try {
      const response = await fetch(`${API_URL}/api/simulations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transactions: 1000, seed: Date.now() % 100_000 }),
      })
      if (!response.ok) throw new Error('Simulation failed')
      await loadDashboard()
      setSelectedId(null)
    } catch {
      setError('The simulation could not run. Check that the backend is online.')
    } finally {
      setRunning(false)
    }
  }

  async function loadRazorpayCheckout() {
    if (window.Razorpay) return
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement('script')
      script.src = 'https://checkout.razorpay.com/v1/checkout.js'
      script.onload = () => resolve()
      script.onerror = () => reject(new Error('Razorpay Checkout failed to load'))
      document.body.appendChild(script)
    })
  }

  async function startTestPayment() {
    setPaymentLoading(true)
    setPaymentMessage('')
    try {
      if (!razorpayStatus?.configured) {
        throw new Error('Add fresh Test Mode credentials to backend/.env first.')
      }
      await loadRazorpayCheckout()
      const response = await fetch(`${API_URL}/api/payments/create-order`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          amount: 249900,
          currency: 'INR',
          customer_name: 'Amit',
          purpose: 'subscription_recovery_test',
        }),
      })
      const order = await response.json()
      if (!response.ok) throw new Error(order.detail ?? 'Order creation failed')
      if (!window.Razorpay) throw new Error('Razorpay Checkout is unavailable')

      const checkout = new window.Razorpay({
        key: order.key_id,
        amount: order.amount,
        currency: order.currency,
        name: 'RecoverAI',
        description: 'Test subscription payment · no real money',
        order_id: order.id,
        handler: () => {
          setPaymentMessage('Test payment succeeded. Waiting for the signed webhook.')
        },
        modal: {
          ondismiss: () => setPaymentMessage('Checkout closed without completing payment.'),
        },
        theme: { color: '#5968ed' },
      })
      checkout.on('payment.failed', () => {
        setPaymentMessage('Test payment failed. The webhook should now persist the failure.')
      })
      checkout.open()
    } catch (paymentError) {
      setPaymentMessage(
        paymentError instanceof Error ? paymentError.message : 'Unable to start Test Checkout.',
      )
    } finally {
      setPaymentLoading(false)
    }
  }

  async function startTestSubscription() {
    setPaymentLoading(true)
    setPaymentMessage('')
    try {
      if (!razorpayStatus?.configured) {
        throw new Error('Add fresh Test Mode credentials to backend/.env first.')
      }
      await loadRazorpayCheckout()
      const response = await fetch(`${API_URL}/api/subscriptions/create-test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          amount: 249900,
          plan_name: 'RecoverAI Pro',
          total_count: 6,
        }),
      })
      const subscription = await response.json()
      if (!response.ok) throw new Error(subscription.detail ?? 'Subscription creation failed')
      if (!window.Razorpay) throw new Error('Razorpay Checkout is unavailable')

      const checkout = new window.Razorpay({
        key: subscription.key_id,
        subscription_id: subscription.subscription_id,
        name: 'RecoverAI',
        description: 'RecoverAI Pro · Test subscription authentication',
        handler: () => {
          setPaymentMessage(
            'Subscription authenticated. Trigger a failed test charge in Razorpay Dashboard.',
          )
        },
        modal: {
          ondismiss: () => setPaymentMessage('Subscription authentication was cancelled.'),
        },
        theme: { color: '#5968ed' },
      })
      checkout.on('payment.failed', () => {
        setPaymentMessage('Authentication payment failed. Check the signed webhook event.')
      })
      checkout.open()
    } catch (subscriptionError) {
      setPaymentMessage(
        subscriptionError instanceof Error
          ? subscriptionError.message
          : 'Unable to start Test Subscription.',
      )
    } finally {
      setPaymentLoading(false)
    }
  }

  useEffect(() => {
    // The initial API request intentionally hydrates dashboard state.
    // oxlint-disable-next-line react/set-state-in-effect
    void loadDashboard()
  }, [])

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">R</span>
          <div>
            <strong>RecoverAI</strong>
            <small>Revenue operations</small>
          </div>
        </div>
        <nav>
          <button className="nav-item active"><span>◈</span> Command center</button>
          <button className="nav-item"><span>◎</span> Recovery queue</button>
          <button className="nav-item"><span>⌁</span> Agent activity</button>
          <button className="nav-item"><span>▥</span> Evaluation</button>
        </nav>
        <div className="policy-card">
          <div className="policy-title"><span className="pulse" /> Policy engine online</div>
          <p>Bounded retries and customer contact limits are enforced.</p>
          <div><span>Violations</span><strong>{metrics?.policy_violations ?? 0}</strong></div>
          <div><span>Duplicate actions</span><strong>{metrics?.duplicate_actions ?? 0}</strong></div>
        </div>
      </aside>

      <main>
        <header>
          <div>
            <p className="eyebrow">MERCHANT CONTROL CENTER</p>
            <h1>Revenue recovery</h1>
            <p className="subtitle">Detect loss, choose a bounded intervention, verify the money.</p>
          </div>
          <button className="run-button" onClick={runSimulation} disabled={running}>
            {running ? <span className="spinner" /> : <span>▶</span>}
            {running ? 'Running agent…' : 'Run 1,000 transactions'}
          </button>
        </header>

        {error && <div className="error-banner">{error}</div>}

        <section className="gateway-card">
          <div className="gateway-copy">
            <div className="gateway-icon">₹</div>
            <div>
              <p className="eyebrow">RAZORPAY TEST MODE</p>
              <h2>RecoverAI Demo Subscription</h2>
              <p>Creates a server-side order, opens Checkout, then waits for a signed webhook.</p>
            </div>
          </div>
          <div className="gateway-status">
            <span className={razorpayStatus?.configured ? 'ready' : 'waiting'}>
              <i /> API {razorpayStatus?.configured ? 'connected' : 'needs fresh keys'}
            </span>
            <span className={razorpayStatus?.webhook_configured ? 'ready' : 'waiting'}>
              <i /> Webhook {razorpayStatus?.webhook_configured ? 'secured' : 'needs secret'}
            </span>
          </div>
          <div className="gateway-action">
            <div><span>Test amount</span><strong>₹2,499</strong></div>
            <div className="gateway-buttons">
              <button
                className="secondary-pay-button"
                onClick={startTestPayment}
                disabled={paymentLoading || !razorpayStatus?.configured}
              >
                Test order
              </button>
              <button
                className="pay-button"
                onClick={startTestSubscription}
                disabled={paymentLoading || !razorpayStatus?.configured}
              >
                {paymentLoading ? 'Connecting…' : 'Start subscription'}
              </button>
            </div>
          </div>
          {paymentMessage && <p className="payment-message">{paymentMessage}</p>}
        </section>

        <section className="metric-grid" aria-busy={loading}>
          <article className="metric hero-metric">
            <span>Money recovered</span>
            <strong>{money(metrics?.revenue_recovered ?? 0)}</strong>
            <small>Verified payment outcomes</small>
          </article>
          <article className="metric">
            <span>Revenue at risk</span>
            <strong>{money(metrics?.revenue_at_risk ?? 0)}</strong>
            <small>{metrics?.failed_transactions ?? 0} failed subscriptions</small>
          </article>
          <article className="metric">
            <span>Recovery rate</span>
            <strong>{((metrics?.recovery_rate ?? 0) * 100).toFixed(1)}%</strong>
            <small>{metrics?.recovered_cases ?? 0} successful recoveries</small>
          </article>
          <article className="metric">
            <span>Net recovered</span>
            <strong>{money(metrics?.net_recovered ?? 0)}</strong>
            <small>{money(metrics?.intervention_cost ?? 0)} intervention cost</small>
          </article>
        </section>

        <section className="content-grid">
          <article className="panel queue-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">LIVE WORKFLOW</p>
                <h2>Recovery queue</h2>
              </div>
              <span className="count-chip">{cases.length} shown</span>
            </div>
            <div className="queue-header">
              <span>Customer</span><span>At risk</span><span>Strategy</span><span>Score</span>
            </div>
            <div className="queue-list">
              {cases.map((item) => (
                <button
                  key={item.id}
                  className={`queue-row ${selected?.id === item.id ? 'selected' : ''}`}
                  onClick={() => setSelectedId(item.id)}
                >
                  <span className="customer">
                    <i>{item.customer_name.split(' ').map((word) => word[0]).join('')}</i>
                    <span><strong>{item.customer_name}</strong><small>{item.failure_reason.replaceAll('_', ' ')}</small></span>
                  </span>
                  <strong>{money(item.amount)}</strong>
                  <span className="strategy">{label(item.strategy)}</span>
                  <span className="score">{Math.round(item.recovery_probability * 100)}%</span>
                </button>
              ))}
            </div>
          </article>

          <article className="panel case-panel">
            {selected ? (
              <>
                <div className="case-top">
                  <div>
                    <p className="eyebrow">{selected.id}</p>
                    <h2>{selected.customer_name}</h2>
                  </div>
                  <span className={`status ${selected.status}`}>{label(selected.status)}</span>
                </div>
                <div className="case-facts">
                  <div><span>Amount at risk</span><strong>{money(selected.amount)}</strong></div>
                  <div><span>Recovery score</span><strong>{Math.round(selected.recovery_probability * 100)}%</strong></div>
                  <div><span>Preferred language</span><strong>{selected.language}</strong></div>
                  <div><span>Payment history</span><strong>{selected.successful_payments}/{selected.previous_payments} paid</strong></div>
                </div>
                <div className="recommendation">
                  <span>Selected intervention</span>
                  <strong>{label(selected.strategy)}</strong>
                  <p>Chosen from payment reason, customer reliability, expected value, and prior contacts.</p>
                </div>
                <div className="audit-heading">
                  <h3>Explainable audit trail</h3>
                  <span>{selected.audit.length} events</span>
                </div>
                <div className="timeline">
                  {selected.audit.map((entry, index) => (
                    <div className="timeline-item" key={`${entry.timestamp}-${index}`}>
                      <span className={`timeline-dot ${entry.tone}`} />
                      <time>{new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time>
                      <div><strong>{entry.event}</strong><p>{entry.detail}</p></div>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="empty">Run a simulation to create recovery cases.</div>
            )}
          </article>
        </section>

        <footer>
          <span>Simulation mode · No real money actions</span>
          <span>{metrics?.transactions ?? 0} transactions evaluated · {metrics?.interventions ?? 0} bounded interventions</span>
        </footer>
      </main>
    </div>
  )
}

export default App
