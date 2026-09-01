import { lazy, Suspense, useEffect, useMemo, useState } from 'react'
import './App.css'

const Beams = lazy(() => import('./components/Beams'))

const rawApiUrl =
  import.meta.env.VITE_API_URL ??
  (import.meta.env.DEV ? 'http://localhost:8010' : '')
const API_URL = rawApiUrl.replace(/\/$/, '')

type View =
  | 'overview'
  | 'at_risk'
  | 'recoveries'
  | 'promises'
  | 'interventions'
  | 'inbox'
  | 'failures'
  | 'customers'
  | 'agent'
  | 'analytics'
  | 'automations'
  | 'safety'
  | 'settings'

type RazorpayCheckout = {
  open: () => void
  on: (event: string, handler: (response: unknown) => void) => void
}

declare global {
  interface Window {
    Razorpay?: new (options: Record<string, unknown>) => RazorpayCheckout
  }
}

type QueueCase = {
  id: string
  customer_id: string
  customer_name: string
  amount: number
  language: string
  payment_method: string
  payment_status: string
  failure_reason: string
  merchant_label?: string
  mapped_class?: string
  recoverability?: string
  status: string
  strategy: string
  selected_action: string | null
  selected_action_display?: string
  recovery_probability: number
  agent_message: string | null
  communication_status: string | null
  execution_mode: string
  executed: boolean
  created_at: string
}

type AgentActivityItem = {
  decision_id: string
  case_id: string
  customer_name: string
  amount: number
  action: string | null
  message: string | null
  language: string | null
  communication_status: string | null
  recovery_probability: number | null
  executed: boolean
  created_at: string
}

type LanguageOption = { id: string; label: string }

const FALLBACK_LANGUAGES: LanguageOption[] = [
  { id: 'english', label: 'English' },
  { id: 'hinglish', label: 'Hinglish' },
  { id: 'hindi', label: 'हिन्दी' },
  { id: 'tamil', label: 'தமிழ்' },
  { id: 'telugu', label: 'తెలుగు' },
  { id: 'marathi', label: 'मराठी' },
  { id: 'bengali', label: 'বাংলা' },
  { id: 'gujarati', label: 'ગુજરાતી' },
  { id: 'kannada', label: 'ಕನ್ನಡ' },
  { id: 'malayalam', label: 'മലയാളം' },
  { id: 'punjabi', label: 'ਪੰਜਾਬੀ' },
]

type EvaluationSummary = {
  at_risk_revenue: number
  predicted_recoverable: number
  observed_recovered: number
  open_failed_cases: number
  recovered_cases: number
  shadow_agent_plans: number
  observational_recoveries: number
  attributed_intervention_recoveries: number
  real_actions_executed?: number
  policy_violations?: number
  fraud_actions?: number
  unauthorized_actions?: number
  execution_safety_pct?: number
  execution_mode?: string
  pilot_enabled?: boolean
  kill_switch?: boolean
  recovery_rate: number
  series: Array<{ date: string; at_risk: number; predicted: number; recovered: number }>
  note: string
}

type MerchantCustomer = {
  id: string
  name: string
  email: string | null
  payments: number
  failed: number
  recovered_count: number
  recovered_amount: number
  at_risk_amount: number
  risk: string
}

type InterventionStat = {
  action: string
  cases: number
  at_risk: number
  recovered: number
  recovered_cases: number
  recovery_rate: number
}

type Timeline = {
  found: boolean
  customer_name?: string
  amount?: number
  payment_status?: string
  failure_reason?: string
  action?: string | null
  recovery_probability?: number | null
  agent_message?: string | null
  amount_recovered?: number
  attribution?: string | null
  model_version?: string | null
  policy_version?: string | null
  executed?: boolean
  execution_mode?: string
  diagnosis?: {
    mapped_class: string
    recoverability: string
    merchant_label: string
    recommended_action: string
    auto_retry_allowed: boolean
    policy_checks: string[]
  }
  events: Array<{ at: string | null; event: string; detail: string; tone: string }>
}

type FailureGallery = {
  classes: Array<{
    mapped_class: string
    recoverability: string
    merchant_label: string
    count: number
    at_risk: number
    recovered: number
    examples: string[]
    auto_retry_allowed: boolean
  }>
  incidents: Array<{
    case_id: string
    customer_name: string
    amount: number
    mapped_class: string
    recoverability: string
    merchant_label: string
    recommended_action: string
    policy_checks: string[]
    money_state: string
    graceful_stop: boolean
  }>
  catalog_note?: string
}

type NorthStar = {
  recoverai: {
    recovered_inr: number
    recovered_cases: number
    open_cases: number
    terminal_graceful_stops: number
    customer_repair_paths: number
  }
  baseline: {
    label: string
    would_false_retry: number
  }
  advantage: {
    hard_declines_blocked: number
    false_retries_avoided_est: number
    note: string
  }
}

type RuntimeHealth = {
  redis: string
  worker: string
  database: string
  pending_jobs: number
  failed_jobs: number
}

type RazorpayStatus = {
  configured: boolean
  webhook_configured: boolean
}

type OutcomeMetrics = {
  evidence_inventory?: {
    real_failures_observed: number
    shadow_decisions: number
    attributed_intervention_recoveries: number
    observational_recoveries: number
  }
  recoverai_state?: { state: string; next_action: string }
  shadow_decisions?: number
}

type ShadowMetrics = {
  shadow_decisions: number
  blocked_actions: number
  action_distribution: Record<string, number>
  policy_violations: number
  automated_fraud_actions: number
  duplicate_decisions: number
  predicted_recovery_value: number
}

const money = (value: number) =>
  new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: value % 1 === 0 ? 0 : 2,
  }).format(value)

const label = (value: string) =>
  value.replaceAll('_', ' ').replace(/\b\w/g, (c) => c.toUpperCase())

type InboxPayload = {
  count: number
  total_amount: number
  auto_handling_count: number
  items: Array<{
    id: string
    kind: string
    priority: string
    title: string
    customer_name: string
    amount: number
    reason: string
    case_id: string | null
  }>
  note?: string
}

type PromiseSummary = {
  promised_amount: number
  collected_amount: number
  pending_amount: number
  overdue_amount: number
  pending_count: number
  overdue_count: number
  fulfilled_count: number
  total_count: number
}

type PromiseItem = {
  id: string
  customer_name: string
  amount: number
  note: string | null
  deadline: string | null
  status: string
  language: string
  recovery_case_id: string | null
  reminder_count: number
}

type AutomationStatus = {
  active: boolean
  recover_revenue_enabled: boolean
  kill_switch: boolean
  headline: string
  note: string
  checks: Array<{ id: string; label: string; ok: boolean }>
  mode: string
  max_automatic_recovery_inr: number
}

type CampaignOverview = {
  campaigns: Array<{
    id: string
    name: string
    enabled: boolean
    step_count: number
    steps: Array<{ id: string; at: string; action: string; communication: boolean }>
  }>
  note?: string
}

const NAV = [
  { id: 'overview' as const, group: 'REVBACK', title: 'Overview', icon: '⌂' },
  { id: 'at_risk' as const, group: 'REVBACK', title: 'At Risk', icon: '₹' },
  { id: 'recoveries' as const, group: 'REVBACK', title: 'Recoveries', icon: '↗' },
  { id: 'promises' as const, group: 'REVBACK', title: 'Promises', icon: '◷' },
  { id: 'automations' as const, group: 'REVBACK', title: 'Automations', icon: '⚡' },
  { id: 'customers' as const, group: 'REVBACK', title: 'Customers', icon: '◉' },
  { id: 'analytics' as const, group: 'REVBACK', title: 'Analytics', icon: '▦' },
  { id: 'agent' as const, group: 'REVBACK', title: 'AI Agent', icon: '✦' },
  { id: 'inbox' as const, group: 'REVBACK', title: 'Needs Attention', icon: '⚠' },
  { id: 'safety' as const, group: 'SYSTEM', title: 'Safety & Audit', icon: '⛨' },
  { id: 'settings' as const, group: 'SYSTEM', title: 'Settings', icon: '⚙' },
]

function App() {
  const [view, setView] = useState<View>('overview')
  const [queue, setQueue] = useState<QueueCase[]>([])
  const [activity, setActivity] = useState<AgentActivityItem[]>([])
  const [evaluation, setEvaluation] = useState<EvaluationSummary | null>(null)
  const [customers, setCustomers] = useState<MerchantCustomer[]>([])
  const [interventions, setInterventions] = useState<InterventionStat[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [timeline, setTimeline] = useState<Timeline | null>(null)
  const [runtimeHealth, setRuntimeHealth] = useState<RuntimeHealth | null>(null)
  const [razorpayStatus, setRazorpayStatus] = useState<RazorpayStatus | null>(null)
  const [, setOutcomeMetrics] = useState<OutcomeMetrics | null>(null)
  const [shadowMetrics, setShadowMetrics] = useState<ShadowMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState('')
  const [paymentLoading, setPaymentLoading] = useState(false)
  const [paymentMessage, setPaymentMessage] = useState('')
  const [killBusy, setKillBusy] = useState(false)
  const [customerQuery, setCustomerQuery] = useState('')
  const [languages, setLanguages] = useState<LanguageOption[]>(FALLBACK_LANGUAGES)
  const [draftLanguage, setDraftLanguage] = useState('hinglish')
  const [draftingCaseId, setDraftingCaseId] = useState<string | null>(null)
  const [inbox, setInbox] = useState<InboxPayload | null>(null)
  const [promiseSummary, setPromiseSummary] = useState<PromiseSummary | null>(null)
  const [promises, setPromises] = useState<PromiseItem[]>([])
  const [automation, setAutomation] = useState<AutomationStatus | null>(null)
  const [campaigns, setCampaigns] = useState<CampaignOverview | null>(null)
  const [automationBusy, setAutomationBusy] = useState(false)
  const [failureGallery, setFailureGallery] = useState<FailureGallery | null>(null)
  const [northStar, setNorthStar] = useState<NorthStar | null>(null)
  const [riskFilter, setRiskFilter] = useState<'all' | 'high' | 'attention'>('all')

  const atRiskCases = useMemo(
    () => queue.filter((item) => item.status === 'active' || item.payment_status === 'failed'),
    [queue],
  )
  const recoveredCases = useMemo(
    () => queue.filter((item) => item.status === 'recovered' || ['captured', 'paid', 'recovered'].includes(item.payment_status)),
    [queue],
  )
  const filteredAtRisk = useMemo(() => {
    const attentionIds = new Set((inbox?.items ?? []).map((i) => i.case_id).filter(Boolean))
    if (riskFilter === 'high') return atRiskCases.filter((c) => c.amount >= 5000)
    if (riskFilter === 'attention') return atRiskCases.filter((c) => attentionIds.has(c.id))
    return atRiskCases
  }, [atRiskCases, riskFilter, inbox])
  const actionCounts = useMemo(() => {
    const dist = shadowMetrics?.action_distribution ?? {}
    return {
      links: dist.payment_link ?? 0,
      retries: (dist.retry_payment ?? 0) + (dist.silent_retry ?? 0),
      reminders: (dist.whatsapp_reminder ?? 0) + (dist.send_reminder ?? 0),
      stopped: (dist.do_nothing ?? 0) + (dist.stop ?? 0),
      escalate: dist.escalate_to_merchant ?? 0,
    }
  }, [shadowMetrics])
  const drawerCase = useMemo(() => {
    if (!selectedId) return null
    const pool = view === 'recoveries' ? recoveredCases : view === 'at_risk' ? filteredAtRisk : queue
    return pool.find((item) => item.id === selectedId) ?? queue.find((item) => item.id === selectedId) ?? null
  }, [selectedId, view, queue, filteredAtRisk, recoveredCases])
  const filteredCustomers = useMemo(() => {
    const q = customerQuery.trim().toLowerCase()
    if (!q) return customers
    return customers.filter((c) => c.name.toLowerCase().includes(q) || (c.email ?? '').toLowerCase().includes(q))
  }, [customers, customerQuery])

  const greeting = useMemo(() => {
    const hour = new Date().getHours()
    if (hour < 12) return 'Good morning'
    if (hour < 17) return 'Good afternoon'
    return 'Good evening'
  }, [])

  async function loadDashboard() {
    setRefreshing(true)
    setError('')
    try {
      const [
        queueRes, activityRes, evalRes, customersRes, interventionsRes,
        healthRes, razorpayRes, outcomesRes, shadowRes, languagesRes,
        inboxRes, promisesRes, automationRes, campaignsRes, galleryRes, northStarRes,
      ] = await Promise.all([
        fetch(`${API_URL}/api/recovery/queue?limit=100`),
        fetch(`${API_URL}/api/recovery/agent/activity?limit=100`),
        fetch(`${API_URL}/api/recovery/evaluation`),
        fetch(`${API_URL}/api/recovery/customers?limit=100`),
        fetch(`${API_URL}/api/recovery/interventions/stats`),
        fetch(`${API_URL}/api/recovery/worker/health`),
        fetch(`${API_URL}/api/razorpay/status`),
        fetch(`${API_URL}/api/recovery/outcomes/metrics`),
        fetch(`${API_URL}/api/recovery/shadow/metrics`),
        fetch(`${API_URL}/api/recovery/agent/languages`),
        fetch(`${API_URL}/api/recovery/inbox?limit=20`),
        fetch(`${API_URL}/api/recovery/promises?limit=50`),
        fetch(`${API_URL}/api/recovery/automation`),
        fetch(`${API_URL}/api/recovery/campaigns`),
        fetch(`${API_URL}/api/recovery/failure-gallery?limit=100`),
        fetch(`${API_URL}/api/recovery/north-star`),
      ])
      const failed = [queueRes, activityRes, evalRes, customersRes, interventionsRes, healthRes, razorpayRes, outcomesRes, shadowRes]
        .filter((r) => !r.ok)
      if (failed.length === 9) throw new Error('Backend unavailable on port 8010')
      if (queueRes.ok) {
        const next = (await queueRes.json()) as QueueCase[]
        setQueue(next)
        setSelectedId((cur) => cur ?? next[0]?.id ?? null)
      }
      if (activityRes.ok) setActivity((await activityRes.json()) as AgentActivityItem[])
      if (evalRes.ok) setEvaluation((await evalRes.json()) as EvaluationSummary)
      if (customersRes.ok) setCustomers((await customersRes.json()) as MerchantCustomer[])
      if (interventionsRes.ok) setInterventions((await interventionsRes.json()) as InterventionStat[])
      if (healthRes.ok) setRuntimeHealth((await healthRes.json()) as RuntimeHealth)
      if (razorpayRes.ok) setRazorpayStatus((await razorpayRes.json()) as RazorpayStatus)
      if (outcomesRes.ok) setOutcomeMetrics((await outcomesRes.json()) as OutcomeMetrics)
      if (shadowRes.ok) setShadowMetrics((await shadowRes.json()) as ShadowMetrics)
      if (languagesRes.ok) {
        const payload = (await languagesRes.json()) as { supported?: LanguageOption[]; default?: string }
        if (payload.supported?.length) setLanguages(payload.supported)
        if (payload.default) setDraftLanguage(payload.default)
      }
      if (inboxRes.ok) setInbox((await inboxRes.json()) as InboxPayload)
      if (promisesRes.ok) {
        const payload = (await promisesRes.json()) as { summary: PromiseSummary; items: PromiseItem[] }
        setPromiseSummary(payload.summary)
        setPromises(payload.items)
      }
      if (automationRes.ok) setAutomation((await automationRes.json()) as AutomationStatus)
      if (campaignsRes.ok) setCampaigns((await campaignsRes.json()) as CampaignOverview)
      if (galleryRes.ok) setFailureGallery((await galleryRes.json()) as FailureGallery)
      if (northStarRes.ok) setNorthStar((await northStarRes.json()) as NorthStar)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load RevBack dashboard')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  async function loadTimeline(caseId: string) {
    try {
      const response = await fetch(`${API_URL}/api/recovery/cases/${caseId}/timeline`)
      if (!response.ok) {
        setTimeline(null)
        return
      }
      setTimeline((await response.json()) as Timeline)
    } catch {
      setTimeline(null)
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
      if (!razorpayStatus?.configured) throw new Error('Add Test Mode credentials to backend/.env first.')
      await loadRazorpayCheckout()
      const response = await fetch(`${API_URL}/api/payments/create-order`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          amount: 299900,
          currency: 'INR',
          customer_name: 'Rahul',
          purpose: 'merchant_recovery_demo',
        }),
      })
      const order = await response.json()
      if (!response.ok) throw new Error(order.detail ?? 'Order creation failed')
      if (!window.Razorpay) throw new Error('Razorpay Checkout unavailable')
      const checkout = new window.Razorpay({
        key: order.key_id,
        amount: order.amount,
        currency: order.currency,
        name: 'RevBack',
        description: 'Test Mode recovery payment · ₹2,999',
        order_id: order.id,
        handler: () => setPaymentMessage('Payment succeeded. Waiting for webhook.'),
        modal: { ondismiss: () => setPaymentMessage('Checkout closed without payment.') },
        theme: { color: '#2f6fed' },
      })
      checkout.on('payment.failed', () => {
        setPaymentMessage('Payment failed. RevBack should create an At Risk case.')
        void loadDashboard()
      })
      checkout.open()
    } catch (err) {
      setPaymentMessage(err instanceof Error ? err.message : 'Unable to start checkout')
    } finally {
      setPaymentLoading(false)
    }
  }

  async function toggleKillSwitch(armed: boolean) {
    setKillBusy(true)
    try {
      const response = await fetch(`${API_URL}/api/recovery/kill-switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ armed }),
      })
      if (!response.ok) throw new Error('Kill switch update failed')
      await loadDashboard()
    } catch {
      setError('Could not update kill switch. Is Redis running?')
    } finally {
      setKillBusy(false)
    }
  }

  async function toggleRecoverRevenue(enabled: boolean) {
    setAutomationBusy(true)
    setError('')
    try {
      const response = await fetch(`${API_URL}/api/recovery/automation/recover-revenue`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled }),
      })
      const payload = await response.json()
      if (!response.ok) throw new Error(payload.detail ?? 'Unable to update automation')
      setAutomation(payload as AutomationStatus)
      setPaymentMessage(payload.headline ?? (enabled ? 'RevBack is active.' : 'Automation paused.'))
      await loadDashboard()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Automation update failed')
    } finally {
      setAutomationBusy(false)
    }
  }

  async function createPromiseForCase(caseId: string) {
    try {
      const response = await fetch(`${API_URL}/api/recovery/promises`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          case_id: caseId,
          days: 1,
          note: 'Merchant recorded promise to pay tomorrow',
          language: draftLanguage,
        }),
      })
      const payload = await response.json()
      if (!response.ok || payload.ok === false) throw new Error(payload.reason ?? 'Promise create failed')
      setPaymentMessage(`Promise recorded for ${payload.promise?.customer_name ?? 'customer'}.`)
      await loadDashboard()
      setView('promises')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create promise')
    }
  }

  async function draftInLanguage(caseId: string, language: string) {
    setDraftingCaseId(caseId)
    setError('')
    try {
      const response = await fetch(`${API_URL}/api/recovery/agent/draft-language`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ case_id: caseId, language }),
      })
      const payload = await response.json()
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.blocked_reason ?? payload.detail ?? 'Draft failed')
      }
      setActivity((prev) =>
        prev.map((item) =>
          item.case_id === caseId
            ? {
                ...item,
                message: payload.message ?? item.message,
                language: payload.language ?? language,
                communication_status: payload.communication_status ?? 'drafted',
                action: payload.action ?? item.action,
              }
            : item,
        ),
      )
      setPaymentMessage(
        `Drafted ${languages.find((l) => l.id === language)?.label ?? language} message for ${payload.customer_name ?? 'customer'} (preview only).`,
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to draft language message')
    } finally {
      setDraftingCaseId(null)
    }
  }

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void loadDashboard()
  }, [])

  useEffect(() => {
    if (!selectedId) {
      setTimeline(null)
      return
    }
    // oxlint-disable-next-line react/set-state-in-effect
    void loadTimeline(selectedId)
  }, [selectedId])

  function openCase(caseId: string) {
    setSelectedId(caseId)
    setView('at_risk')
  }

  const healthy =
    runtimeHealth?.redis === 'available'
    && runtimeHealth?.worker === 'ready'
    && (runtimeHealth?.database === 'postgresql' || Boolean(runtimeHealth?.database))

  const navGroups = useMemo(() => {
    const groups: Array<{ name: string; items: typeof NAV }> = []
    for (const item of NAV) {
      const existing = groups.find((g) => g.name === item.group)
      if (existing) existing.items.push(item)
      else groups.push({ name: item.group, items: [item] })
    }
    return groups
  }, [])

  return (
    <div className="app-root">
      <div className="beams-bg" aria-hidden="true">
        <Suspense fallback={null}>
          <Beams
            beamWidth={2.2}
            beamHeight={16}
            beamNumber={14}
            lightColor="#3395FF"
            beamColor="#0A2148"
            backgroundColor="#041530"
            speed={1.6}
            noiseIntensity={1.4}
            scale={0.18}
            rotation={28}
          />
        </Suspense>
      </div>

    <div className="merchant-shell">
      <aside className="merchant-sidebar">
        <div className="brand-block">
          <img
            src="/recoverai-logo.png"
            alt="RevBack"
            className="brand-logo"
          />
          <div>
            <strong>RevBack</strong>
            <small>Detect · Recover · Revenue</small>
          </div>
        </div>

        <nav className="merchant-nav">
          {navGroups.map((group) => (
            <div key={group.name} className="nav-group">
              <p className="nav-group-label">{group.name}</p>
              {group.items.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`nav-link ${view === item.id ? 'active' : ''}`}
                  onClick={() => setView(item.id)}
                >
                  <span>{item.icon}</span>
                  {item.title}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-foot">
          <div className={`health-pill ${healthy ? 'ok' : 'bad'}`}>
            <i /> {healthy ? 'System healthy' : 'Needs attention'}
          </div>
          <div className="merchant-chip">
            <strong>Tanish Store</strong>
            <span>Razorpay Test Mode</span>
          </div>
        </div>
      </aside>

      <div className="merchant-main">
        <header className="merchant-topbar">
          <div>
            <h1>
              {view === 'overview' && 'Overview'}
              {view === 'at_risk' && 'At Risk'}
              {view === 'recoveries' && 'Recoveries'}
              {view === 'promises' && 'Promises'}
              {view === 'inbox' && 'Needs Attention'}
              {view === 'customers' && 'Customers'}
              {view === 'agent' && 'AI Agent'}
              {view === 'analytics' && 'Analytics'}
              {view === 'automations' && 'Automations'}
              {view === 'safety' && 'Safety & Audit'}
              {view === 'settings' && 'Settings'}
              {view === 'interventions' && 'Interventions'}
              {view === 'failures' && 'Failure Gallery'}
            </h1>
            <p>
              {view === 'overview' && `${greeting} — RevBack is protecting your revenue.`}
              {view === 'at_risk' && 'Payments RevBack is monitoring automatically.'}
              {view === 'recoveries' && 'Money recovered — observed vs AI-attributed stay honest.'}
              {view === 'promises' && 'Customers who said they will pay — tracked automatically.'}
              {view === 'inbox' && 'Only exceptions. Everything else stays automated.'}
              {view === 'customers' && 'Customer memory across failed and recovered payments.'}
              {view === 'agent' && 'Communication drafts for already-decided actions.'}
              {view === 'analytics' && 'Revenue impact and RevBack vs dumb retry.'}
              {view === 'automations' && 'Turn on Recover Revenue — RevBack handles the rest.'}
              {view === 'safety' && 'Kill switch, blocks, and explainable audit.'}
              {view === 'settings' && 'Business, Razorpay, and developer tools.'}
              {view === 'interventions' && 'Intervention mix'}
              {view === 'failures' && 'Decline classes (demo)'}
            </p>
          </div>
          <div className="top-actions">
            <button type="button" className="ghost-btn" onClick={() => void loadDashboard()} disabled={refreshing}>
              {refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
            {automation?.active ? (
              <button
                type="button"
                className="ghost-btn"
                disabled={automationBusy}
                onClick={() => void toggleRecoverRevenue(false)}
              >
                Pause automation
              </button>
            ) : (
              <button
                type="button"
                className="primary-btn"
                disabled={automationBusy}
                onClick={() => void toggleRecoverRevenue(true)}
              >
                {automationBusy ? 'Starting…' : '⚡ Recover Revenue'}
              </button>
            )}
          </div>
        </header>

        {error && <div className="banner error">{error}</div>}
        {paymentMessage && <div className="banner info">{paymentMessage}</div>}
        {(evaluation?.execution_mode ?? 'shadow') === 'shadow' && (
          <div className="banner warn">
            Shadow mode — RevBack decides and drafts messages, but does not move money yet.
            Attributed recoveries: {evaluation?.attributed_intervention_recoveries ?? 0}.
          </div>
        )}

        {view === 'overview' && (
          <section className="page" aria-busy={loading}>
            <article className={`hero-recover ${automation?.active ? 'on' : ''}`}>
              <div>
                <p className="eyebrow">{automation?.active ? 'AUTOMATION ON' : 'START HERE'}</p>
                <h2>
                  {automation?.active
                    ? 'RevBack is automatically handling failed payments'
                    : 'Turn on Recover Revenue'}
                </h2>
                <p>
                  {automation?.active
                    ? `Monitoring ${atRiskCases.length} open failures. You only handle exceptions.`
                    : 'Connect once, click Recover Revenue, and RevBack runs the recovery loop. You only see money and exceptions.'}
                </p>
                <div className="hero-actions">
                  {automation?.active ? (
                    <button type="button" className="primary-btn on-glow" disabled>
                      Recover Revenue — ON
                    </button>
                  ) : (
                    <button type="button" className="primary-btn" disabled={automationBusy} onClick={() => void toggleRecoverRevenue(true)}>
                      Recover Revenue
                    </button>
                  )}
                  {automation?.active && (
                    <button type="button" className="ghost-btn" disabled={automationBusy} onClick={() => void toggleRecoverRevenue(false)}>
                      Pause
                    </button>
                  )}
                  <button type="button" className="ghost-btn" onClick={() => setView('inbox')}>
                    Needs attention ({inbox?.count ?? 0})
                  </button>
                </div>
              </div>
              <div className="hero-stats">
                <div><span>At risk</span><strong>{money(evaluation?.at_risk_revenue ?? 0)}</strong></div>
                <div><span>Observed recovered</span><strong>{money(evaluation?.observed_recovered ?? 0)}</strong></div>
                <div><span>AI-attributed</span><strong>{evaluation?.attributed_intervention_recoveries ?? 0}</strong></div>
              </div>
            </article>

            <div className="kpi-row">
              <article className="kpi-card danger">
                <span>Revenue at risk</span>
                <strong>{money(evaluation?.at_risk_revenue ?? 0)}</strong>
                <small>{evaluation?.open_failed_cases ?? 0} open failures</small>
              </article>
              <article className="kpi-card success">
                <span>Recovered (observed)</span>
                <strong>{money(evaluation?.observed_recovered ?? 0)}</strong>
                <small>Not claimed as AI-attributed until pilot evidence exists</small>
              </article>
              <article className="kpi-card">
                <span>Recovery rate</span>
                <strong>{((evaluation?.recovery_rate ?? 0) * 100).toFixed(1)}%</strong>
                <small>Observed recoveries / resolved cases</small>
              </article>
              <article className="kpi-card">
                <span>Need you</span>
                <strong>{inbox?.count ?? 0}</strong>
                <small>{money(inbox?.total_amount ?? 0)} in exceptions</small>
              </article>
            </div>

            <div className="split-2">
              <article className="panel">
                <div className="panel-head">
                  <div>
                    <p className="eyebrow">REVBACK IS WORKING</p>
                    <h2>Today&apos;s automatic activity</h2>
                  </div>
                </div>
                <ul className="work-list">
                  <li><strong>{queue.length}</strong> failed payments detected</li>
                  <li><strong>{inbox?.auto_handling_count ?? atRiskCases.length}</strong> automatically handled</li>
                  <li><strong>{actionCounts.links}</strong> payment links planned</li>
                  <li><strong>{actionCounts.retries}</strong> retries planned</li>
                  <li><strong>{actionCounts.reminders}</strong> reminders planned</li>
                  <li><strong>{actionCounts.stopped}</strong> stopped safely</li>
                  <li><strong>{inbox?.count ?? 0}</strong> need your attention</li>
                </ul>
                <ol className="funnel tight">
                  <li><strong>{money(evaluation?.at_risk_revenue ?? 0)}</strong><span>failed / at risk</span></li>
                  <li><strong>{money(evaluation?.predicted_recoverable ?? 0)}</strong><span>AI evaluated</span></li>
                  <li><strong>{shadowMetrics?.shadow_decisions ?? 0}</strong><span>interventions decided</span></li>
                  <li><strong>{money(evaluation?.observed_recovered ?? 0)}</strong><span>observed recovered</span></li>
                </ol>
              </article>

              <article className="panel">
                <div className="panel-head">
                  <div>
                    <p className="eyebrow">YOU NEED TO KNOW</p>
                    <h2>Exceptions only</h2>
                  </div>
                  <button type="button" className="ghost-btn" onClick={() => setView('inbox')}>Open inbox</button>
                </div>
                <div className="agent-feed">
                  {(inbox?.items ?? []).slice(0, 5).map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      className="agent-item attention-item"
                      onClick={() => item.case_id && openCase(item.case_id)}
                    >
                      <div>
                        <strong>{money(item.amount)} — {item.title}</strong>
                        <small>{item.customer_name} · {label(item.priority)}</small>
                      </div>
                      <p>{item.reason}</p>
                    </button>
                  ))}
                  {(inbox?.count ?? 0) === 0 && (
                    <div className="empty">Inbox clear. RevBack is handling open cases automatically.</div>
                  )}
                </div>
              </article>
            </div>
          </section>
        )}

        {(view === 'at_risk' || view === 'recoveries') && (
          <section className="page">
            {view === 'recoveries' && (
              <div className="kpi-row">
                <article className="kpi-card success">
                  <span>Recovered (observed)</span>
                  <strong>{money(evaluation?.observed_recovered ?? 0)}</strong>
                </article>
                <article className="kpi-card danger">
                  <span>At risk</span>
                  <strong>{money(evaluation?.at_risk_revenue ?? 0)}</strong>
                </article>
                <article className="kpi-card">
                  <span>Recovery rate</span>
                  <strong>{((evaluation?.recovery_rate ?? 0) * 100).toFixed(1)}%</strong>
                </article>
                <article className="kpi-card">
                  <span>Attributed to RevBack</span>
                  <strong>{evaluation?.attributed_intervention_recoveries ?? 0}</strong>
                  <small>Predicted ≠ recovered</small>
                </article>
              </div>
            )}
            <div className={`split-case ${drawerCase ? 'open' : ''}`}>
            <article className="panel list-panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">AUTOMATIC QUEUE</p>
                  <h2>{view === 'at_risk' ? 'At risk — RevBack monitoring' : 'Recovered payments'}</h2>
                </div>
                <span className="count">{view === 'at_risk' ? filteredAtRisk.length : recoveredCases.length}</span>
              </div>
              {view === 'at_risk' && (
                <div className="filter-chips">
                  {([
                    ['all', 'All'],
                    ['high', 'High value'],
                    ['attention', 'Needs attention'],
                  ] as const).map(([id, title]) => (
                    <button
                      key={id}
                      type="button"
                      className={`lang-chip ${riskFilter === id ? 'on' : ''}`}
                      onClick={() => setRiskFilter(id)}
                    >
                      {title}
                    </button>
                  ))}
                </div>
              )}
              <div className="table">
                <div className="table-head">
                  <span>Customer</span><span>Amount</span><span>Problem</span><span>RevBack</span>
                </div>
                {(view === 'at_risk' ? filteredAtRisk : recoveredCases).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={`table-row ${selectedId === item.id ? 'selected' : ''}`}
                    onClick={() => setSelectedId(item.id)}
                  >
                    <span>
                      <strong>{item.customer_name}</strong>
                      <small className={`status-dot ${item.status}`}>{label(item.status === 'active' ? 'recovering' : item.status)}</small>
                    </span>
                    <span>{money(item.amount)}</span>
                    <span>{item.merchant_label ?? label(item.failure_reason)}</span>
                    <span className="pill blue">{label(item.selected_action_display ?? item.selected_action ?? item.strategy)}</span>
                  </button>
                ))}
                {(view === 'at_risk' ? filteredAtRisk : recoveredCases).length === 0 && (
                  <div className="empty">
                    {view === 'at_risk' ? 'No open at-risk cases.' : 'No recovered payments yet — attributed recoveries stay at 0 until a real payment-link pilot completes.'}
                  </div>
                )}
              </div>
            </article>

            <article className="panel case-drawer">
              {drawerCase && timeline?.found ? (
                <>
                  <div className="drawer-top">
                    <div>
                      <p className="eyebrow">{drawerCase.id.slice(0, 8)}…</p>
                      <h2>{drawerCase.customer_name}</h2>
                      <p>{money(drawerCase.amount)} · {label(drawerCase.failure_reason)}</p>
                    </div>
                    <span className={`status ${drawerCase.status}`}>{label(drawerCase.status)}</span>
                  </div>

                  <div className="drawer-grid">
                    <div><span>Why it happened</span><strong>{timeline.diagnosis?.merchant_label ?? label(timeline.failure_reason ?? drawerCase.failure_reason)}</strong></div>
                    <div><span>Recoverability</span><strong>{label(timeline.diagnosis?.recoverability ?? 'unknown')}</strong></div>
                    <div><span>Recovery probability</span><strong>{timeline.recovery_probability == null ? '—' : `${Math.round(timeline.recovery_probability * 100)}%`}</strong></div>
                    <div><span>Recommended action</span><strong>{label(timeline.diagnosis?.recommended_action ?? timeline.action ?? drawerCase.selected_action ?? drawerCase.strategy)}</strong></div>
                    <div><span>Auto-retry</span><strong>{timeline.diagnosis?.auto_retry_allowed ? 'Allowed' : 'Blocked'}</strong></div>
                    <div><span>Recovered</span><strong>{money(timeline.amount_recovered ?? 0)}</strong></div>
                  </div>

                  {(timeline.diagnosis?.policy_checks?.length ?? 0) > 0 && (
                    <div className="policy-checks">
                      {timeline.diagnosis!.policy_checks.map((check) => (
                        <span key={check} className="pill blue">{label(check)}</span>
                      ))}
                    </div>
                  )}

                  <div className="shadow-chip">
                    {(timeline.execution_mode ?? 'shadow').toUpperCase()} — {timeline.executed ? 'EXECUTED' : 'NOT EXECUTED'}
                  </div>

                  <div className="drawer-actions">
                    <button type="button" className="ghost-btn compact" onClick={() => void createPromiseForCase(drawerCase.id)}>
                      Record promise to pay
                    </button>
                    <button
                      type="button"
                      className="ghost-btn compact"
                      disabled={draftingCaseId === drawerCase.id}
                      onClick={() => void draftInLanguage(drawerCase.id, draftLanguage)}
                    >
                      Draft message
                    </button>
                  </div>

                  {timeline.agent_message && (
                    <div className="message-card">
                      <span>Customer communication</span>
                      <p>{timeline.agent_message}</p>
                    </div>
                  )}

                  <div className="journey-head">
                    <p className="eyebrow">RECOVERY JOURNEY</p>
                    <strong>RevBack is handling this automatically</strong>
                  </div>
                  <div className="timeline">
                    {timeline.events.map((entry, index) => (
                      <div key={`${entry.at}-${index}`} className="timeline-item">
                        <span className={`dot ${entry.tone}`} />
                        <time>{entry.at ? new Date(entry.at).toLocaleTimeString() : '--'}</time>
                        <div><strong>{entry.event}</strong><p>{entry.detail}</p></div>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="empty">Select a case to open the recovery journey.</div>
              )}
            </article>
            </div>
          </section>
        )}

        {view === 'interventions' && (
          <section className="page">
            <div className="card-grid">
              {interventions.map((item) => (
                <article key={item.action} className="metric-tile">
                  <span>{label(item.action)}</span>
                  <strong>{item.cases} cases</strong>
                  <p>{money(item.at_risk)} at risk</p>
                  <p>{item.recovered_cases} recovered · {(item.recovery_rate * 100).toFixed(0)}%</p>
                  <div className="progress"><i style={{ width: `${Math.max(4, item.recovery_rate * 100)}%` }} /></div>
                </article>
              ))}
              {interventions.length === 0 && <div className="empty">No intervention decisions yet.</div>}
            </div>
          </section>
        )}

        {view === 'promises' && (
          <section className="page">
            <div className="kpi-row">
              <article className="kpi-card"><span>Promised</span><strong>{money(promiseSummary?.promised_amount ?? 0)}</strong></article>
              <article className="kpi-card success"><span>Collected</span><strong>{money(promiseSummary?.collected_amount ?? 0)}</strong></article>
              <article className="kpi-card"><span>Pending</span><strong>{money(promiseSummary?.pending_amount ?? 0)}</strong></article>
              <article className="kpi-card danger"><span>Overdue</span><strong>{money(promiseSummary?.overdue_amount ?? 0)}</strong></article>
            </div>
            <article className="panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">PROMISE TO PAY</p>
                  <h2>Tracked commitments</h2>
                </div>
                <span className="count">{promises.length}</span>
              </div>
              <div className="table">
                <div className="table-head wide">
                  <span>Customer</span><span>Amount</span><span>Due</span><span>Reminders</span><span>Status</span>
                </div>
                {promises.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className="table-row wide"
                    onClick={() => item.recovery_case_id && openCase(item.recovery_case_id)}
                  >
                    <span>
                      <strong>{item.customer_name}</strong>
                      <small>{item.note ?? 'Promise to pay'}</small>
                    </span>
                    <span>{money(item.amount)}</span>
                    <span>{item.deadline ? new Date(item.deadline).toLocaleString() : '—'}</span>
                    <span>{item.reminder_count}</span>
                    <span className={`status ${item.status === 'fulfilled' ? 'recovered' : item.status === 'overdue' ? 'stopped' : 'active'}`}>
                      {label(item.status)}
                    </span>
                  </button>
                ))}
                {promises.length === 0 && <div className="empty">No promises yet. Record one from a case drawer.</div>}
              </div>
            </article>
          </section>
        )}

        {view === 'inbox' && (
          <section className="page">
            <div className="kpi-row">
              <article className="kpi-card danger"><span>Needs you</span><strong>{inbox?.count ?? 0}</strong></article>
              <article className="kpi-card"><span>At stake</span><strong>{money(inbox?.total_amount ?? 0)}</strong></article>
              <article className="kpi-card success"><span>On autopilot</span><strong>{inbox?.auto_handling_count ?? 0}</strong></article>
              <article className="kpi-card"><span>Automation</span><strong>{automation?.active ? 'ON' : 'OFF'}</strong></article>
            </div>
            <article className="panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">MERCHANT INBOX</p>
                  <h2>Exceptions only</h2>
                </div>
              </div>
              <div className="agent-feed">
                {(inbox?.items ?? []).map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className="agent-item attention-item"
                    onClick={() => item.case_id && openCase(item.case_id)}
                  >
                    <div>
                      <strong>{item.title}</strong>
                      <small>{item.customer_name} · {money(item.amount)} · {label(item.priority)}</small>
                    </div>
                    <p>{item.reason}</p>
                  </button>
                ))}
                {(inbox?.count ?? 0) === 0 && <div className="empty">Inbox clear. RevBack is handling the rest.</div>}
              </div>
            </article>
          </section>
        )}

        {view === 'failures' && (
          <section className="page">
            <p className="muted-note">{failureGallery?.catalog_note}</p>
            <div className="card-grid">
              {(failureGallery?.classes ?? []).map((cls) => (
                <article key={cls.mapped_class} className="metric-tile">
                  <span>{cls.merchant_label}</span>
                  <strong>{cls.count} cases</strong>
                  <p>{label(cls.recoverability)} · {cls.auto_retry_allowed ? 'retry ok' : 'no blind retry'}</p>
                  <p>{money(cls.at_risk)} at risk · {money(cls.recovered)} recovered</p>
                  <div className="progress">
                    <i style={{ width: `${Math.max(4, (cls.recovered / Math.max(1, cls.at_risk + cls.recovered)) * 100)}%` }} />
                  </div>
                </article>
              ))}
              {(failureGallery?.classes?.length ?? 0) === 0 && (
                <div className="empty">No failure classes yet — create a test failure in Settings.</div>
              )}
            </div>
            <article className="panel" style={{ marginTop: 16 }}>
              <div className="panel-head">
                <div>
                  <p className="eyebrow">INCIDENTS</p>
                  <h2>Mapped declines</h2>
                </div>
              </div>
              <div className="table">
                <div className="table-head wide">
                  <span>Customer</span><span>Class</span><span>Recoverability</span><span>Action</span><span>State</span>
                </div>
                {(failureGallery?.incidents ?? []).slice(0, 40).map((row) => (
                  <button
                    key={row.case_id}
                    type="button"
                    className="table-row wide"
                    onClick={() => openCase(row.case_id)}
                  >
                    <span><strong>{row.customer_name}</strong><small>{money(row.amount)}</small></span>
                    <span>{row.merchant_label}</span>
                    <span>{label(row.recoverability)}</span>
                    <span>{label(row.recommended_action)}</span>
                    <span className={`status ${row.money_state === 'recovered' ? 'recovered' : row.graceful_stop ? 'stopped' : 'active'}`}>
                      {row.graceful_stop ? 'Stopped' : label(row.money_state)}
                    </span>
                  </button>
                ))}
              </div>
            </article>
          </section>
        )}

        {view === 'customers' && (
          <section className="page">
            <article className="panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">CUSTOMER MEMORY</p>
                  <h2>Customers</h2>
                </div>
                <input
                  className="search"
                  placeholder="Search customer…"
                  value={customerQuery}
                  onChange={(e) => setCustomerQuery(e.target.value)}
                />
              </div>
              <div className="table">
                <div className="table-head wide">
                  <span>Customer</span><span>Payments</span><span>Failed</span><span>Recovered</span><span>Risk</span>
                </div>
                {filteredCustomers.map((customer) => (
                  <div key={customer.id} className="table-row static wide">
                    <span><strong>{customer.name}</strong><small>{customer.email ?? 'No email'}</small></span>
                    <span>{customer.payments}</span>
                    <span>{customer.failed}</span>
                    <span>{money(customer.recovered_amount)}</span>
                    <span className={`risk ${customer.risk}`}>{label(customer.risk)}</span>
                  </div>
                ))}
              </div>
            </article>
          </section>
        )}

        {view === 'agent' && (
          <section className="page">
            <div className="split-2">
              <article className="panel">
                <div className="panel-head">
                  <div>
                    <p className="eyebrow">AI RECOVERY AGENT</p>
                    <h2>Active · communication only</h2>
                  </div>
                  <span className="live-dot">Live</span>
                </div>
                <div className="kpi-row compact">
                  <div><span>Cases analyzed</span><strong>{shadowMetrics?.shadow_decisions ?? 0}</strong></div>
                  <div><span>Messages drafted</span><strong>{activity.filter((a) => a.message).length}</strong></div>
                  <div><span>Unsafe actions</span><strong>{evaluation?.unauthorized_actions ?? 0}</strong></div>
                  <div><span>Policy violations</span><strong>{shadowMetrics?.policy_violations ?? 0}</strong></div>
                </div>
                <div className="language-panel">
                  <div className="panel-head tight">
                    <div>
                      <p className="eyebrow">CUSTOMER LANGUAGE</p>
                      <h2>Draft in Indian languages</h2>
                    </div>
                  </div>
                  <p className="muted-note">
                    Pick a language, then draft on any case below. Supports Hinglish + major regional languages.
                    Messages stay preview-only in shadow mode.
                  </p>
                  <div className="lang-chips">
                    {languages.map((lang) => (
                      <button
                        key={lang.id}
                        type="button"
                        className={`lang-chip ${draftLanguage === lang.id ? 'on' : ''}`}
                        onClick={() => setDraftLanguage(lang.id)}
                      >
                        {lang.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="rules-grid">
                  <div>
                    <h3>The agent can</h3>
                    <ul>
                      <li>Draft customer messages</li>
                      <li>Personalize Hinglish / regional copy</li>
                      <li>Explain recovery actions</li>
                      <li>Switch language per customer</li>
                    </ul>
                  </div>
                  <div>
                    <h3>The agent cannot</h3>
                    <ul>
                      <li>Change payment amount</li>
                      <li>Choose financial actions</li>
                      <li>Invent payment links</li>
                      <li>Override safety policy</li>
                      <li>Send without merchant review</li>
                    </ul>
                  </div>
                </div>
              </article>
              <article className="panel">
                <div className="panel-head">
                  <div>
                    <p className="eyebrow">ACTIVITY</p>
                    <h2>Recent plans</h2>
                  </div>
                  <span className="pill blue">
                    {languages.find((l) => l.id === draftLanguage)?.label ?? draftLanguage}
                  </span>
                </div>
                <div className="agent-feed">
                  {activity.slice(0, 8).map((item) => (
                    <div key={item.decision_id} className="agent-item">
                      <div className="agent-item-top">
                        <div>
                          <strong>{item.customer_name}</strong>
                          <small>
                            {money(item.amount)} · {label(item.action ?? 'none')}
                            {item.language ? ` · ${label(item.language)}` : ''}
                          </small>
                        </div>
                        <button
                          type="button"
                          className="ghost-btn compact"
                          disabled={draftingCaseId === item.case_id}
                          onClick={() => void draftInLanguage(item.case_id, draftLanguage)}
                        >
                          {draftingCaseId === item.case_id
                            ? 'Drafting…'
                            : `Draft in ${languages.find((l) => l.id === draftLanguage)?.label ?? draftLanguage}`}
                        </button>
                      </div>
                      <p>{item.message ?? 'No customer message yet — pick a language and draft one.'}</p>
                    </div>
                  ))}
                </div>
              </article>
            </div>
          </section>
        )}

        {view === 'analytics' && (
          <section className="page">
            <div className="kpi-row">
              <article className="kpi-card danger"><span>Revenue at risk</span><strong>{money(evaluation?.at_risk_revenue ?? 0)}</strong></article>
              <article className="kpi-card success"><span>Observed recovered</span><strong>{money(evaluation?.observed_recovered ?? 0)}</strong></article>
              <article className="kpi-card"><span>Recovery rate</span><strong>{((evaluation?.recovery_rate ?? 0) * 100).toFixed(1)}%</strong></article>
              <article className="kpi-card"><span>AI-attributed</span><strong>{evaluation?.attributed_intervention_recoveries ?? 0}</strong><small>Stays 0 until real pilot evidence</small></article>
            </div>
            <div className="split-2">
              <article className="panel">
                <div className="panel-head">
                  <div>
                    <p className="eyebrow">REVBACK VS DUMB RETRY</p>
                    <h2>Why automation beats blind retry</h2>
                  </div>
                </div>
                <div className="drawer-grid">
                  <div><span>Observed recovered</span><strong>{money(northStar?.recoverai.recovered_inr ?? 0)}</strong></div>
                  <div><span>Hard declines blocked</span><strong>{northStar?.advantage.hard_declines_blocked ?? 0}</strong></div>
                  <div><span>False retries avoided</span><strong>{northStar?.advantage.false_retries_avoided_est ?? 0}</strong></div>
                  <div><span>Customer repair paths</span><strong>{northStar?.recoverai.customer_repair_paths ?? 0}</strong></div>
                </div>
                <p className="muted-note">{northStar?.advantage.note}</p>
              </article>
              <article className="panel">
                <div className="panel-head"><div><p className="eyebrow">FUNNEL</p><h2>Revenue path</h2></div></div>
                <ol className="funnel">
                  <li><strong>{queue.length}</strong><span>failed payments</span></li>
                  <li><strong>{shadowMetrics?.shadow_decisions ?? 0}</strong><span>AI evaluated</span></li>
                  <li><strong>{actionCounts.links + actionCounts.retries + actionCounts.reminders}</strong><span>interventions planned</span></li>
                  <li><strong>{recoveredCases.length}</strong><span>recovered</span></li>
                </ol>
              </article>
            </div>
          </section>
        )}

        {view === 'automations' && (
          <section className="page">
            <article className="panel form-panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">RECOVER REVENUE</p>
                  <h2>{automation?.active ? 'ON — RevBack manages failed payments' : 'OFF — click to start'}</h2>
                </div>
                {automation?.active ? (
                  <button type="button" className="ghost-btn" disabled={automationBusy} onClick={() => void toggleRecoverRevenue(false)}>Pause</button>
                ) : (
                  <button type="button" className="primary-btn" disabled={automationBusy} onClick={() => void toggleRecoverRevenue(true)}>Enable</button>
                )}
              </div>
              <p className="muted-note">
                Merchant job: connect Razorpay → click Recover Revenue → done. RevBack diagnoses, decides, drafts messages, tracks promises, and only escalates exceptions.
              </p>
            </article>

            <div className="split-2">
              <article className="panel">
                <div className="panel-head"><div><p className="eyebrow">STRATEGY</p><h2>How RevBack routes failures</h2></div></div>
                <ul className="work-list">
                  <li>Temporary bank / network → Retry automatically</li>
                  <li>Insufficient funds / generic fail → Payment link</li>
                  <li>Customer promised → Reminder sequence</li>
                  <li>Repeated / repair needed → Link + reminder</li>
                  <li>Hard decline / fraud → Stop + review</li>
                  <li>True unknown high-risk → Merchant attention</li>
                </ul>
              </article>
              <article className="panel">
                <div className="panel-head"><div><p className="eyebrow">STOPPING RULES</p><h2>Safety the merchant can understand</h2></div></div>
                <ul className="work-list">
                  <li>Max contacts / attempt caps enforced</li>
                  <li>Cooldown between retries</li>
                  <li>Never contact after successful payment</li>
                  <li>Never act on fraud cases</li>
                  <li>Never change payment amount</li>
                  <li>Hard declines never auto-retried</li>
                </ul>
              </article>
            </div>

            <article className="panel" style={{ marginTop: 16 }}>
              <div className="panel-head">
                <div>
                  <p className="eyebrow">COMMUNICATION</p>
                  <h2>Automatic customer messages</h2>
                </div>
              </div>
              <div className="lang-chips" style={{ padding: '0 18px 12px' }}>
                {languages.slice(0, 8).map((lang) => (
                  <span key={lang.id} className="lang-chip on">{lang.label}</span>
                ))}
              </div>
              <p className="muted-note">Qwen / templates only write the message. Policy still chooses payment link vs retry vs escalate.</p>
            </article>

            <article className="panel" style={{ marginTop: 16 }}>
              <div className="panel-head">
                <div>
                  <p className="eyebrow">CAMPAIGNS</p>
                  <h2>Recovery sequences</h2>
                </div>
              </div>
              <div className="card-grid">
                {(campaigns?.campaigns ?? []).map((camp) => (
                  <article key={camp.id} className="metric-tile">
                    <span>{camp.enabled ? 'ON' : 'OFF'}</span>
                    <strong>{camp.name}</strong>
                    <p>{camp.step_count} steps</p>
                    <ul className="mini-steps">
                      {camp.steps.map((step) => (
                        <li key={step.id}>{step.at}: {label(step.action)}</li>
                      ))}
                    </ul>
                  </article>
                ))}
              </div>
            </article>
          </section>
        )}

        {view === 'safety' && (
          <section className="page">
            <div className="kpi-row">
              <article className="kpi-card"><span>Blocked actions</span><strong>{shadowMetrics?.blocked_actions ?? 0}</strong></article>
              <article className="kpi-card"><span>Fraud blocks</span><strong>{shadowMetrics?.automated_fraud_actions ?? 0}</strong></article>
              <article className="kpi-card"><span>Policy violations</span><strong>{shadowMetrics?.policy_violations ?? 0}</strong></article>
              <article className="kpi-card"><span>Duplicates</span><strong>{shadowMetrics?.duplicate_decisions ?? 0}</strong></article>
            </div>
            <div className="split-2">
              <article className="panel">
                <div className="panel-head">
                  <div>
                    <p className="eyebrow">SAFETY STATUS</p>
                    <h2>{evaluation?.kill_switch ? 'Execution blocked' : 'All systems protected'}</h2>
                  </div>
                  <button
                    type="button"
                    className="danger-btn"
                    disabled={killBusy}
                    onClick={() => void toggleKillSwitch(!(evaluation?.kill_switch ?? false))}
                  >
                    {evaluation?.kill_switch ? 'Clear emergency stop' : 'Emergency stop'}
                  </button>
                </div>
                <div className="drawer-grid">
                  <div><span>Kill switch</span><strong>{evaluation?.kill_switch ? 'ARMED' : 'OFF'}</strong></div>
                  <div><span>Execution mode</span><strong>{(evaluation?.execution_mode ?? 'shadow').toUpperCase()}</strong></div>
                  <div><span>Pilot</span><strong>{(evaluation?.pilot_enabled ?? false) ? 'ON' : 'OFF'}</strong></div>
                  <div><span>Safety score</span><strong>{(evaluation?.execution_safety_pct ?? 100).toFixed(0)}%</strong></div>
                </div>
              </article>
              <article className="panel">
                <div className="panel-head"><div><p className="eyebrow">AUDIT</p><h2>Recent decisions</h2></div></div>
                <div className="agent-feed">
                  {activity.slice(0, 6).map((item) => (
                    <div key={item.decision_id} className="agent-item">
                      <div>
                        <strong>{item.customer_name}</strong>
                        <small>{new Date(item.created_at).toLocaleString()}</small>
                      </div>
                      <p>
                        {label(item.action ?? 'none')} · executed {item.executed ? 'yes' : 'no'} ·{' '}
                        {item.communication_status ?? 'n/a'}
                      </p>
                    </div>
                  ))}
                </div>
              </article>
            </div>
          </section>
        )}

        {view === 'settings' && (
          <section className="page">
            <div className="split-2">
              <article className="panel form-panel">
                <div className="panel-head"><div><p className="eyebrow">BUSINESS</p><h2>Store profile</h2></div></div>
                <div className="form-grid">
                  <label><span>Business name</span><input disabled value="Tanish Store" /></label>
                  <label><span>Currency</span><input disabled value="INR" /></label>
                  <label><span>Timezone</span><input disabled value="Asia/Kolkata" /></label>
                </div>
              </article>
              <article className="panel form-panel">
                <div className="panel-head"><div><p className="eyebrow">RAZORPAY</p><h2>Connection</h2></div></div>
                <div className="drawer-grid">
                  <div><span>API</span><strong>{razorpayStatus?.configured ? 'Connected' : 'Needs keys'}</strong></div>
                  <div><span>Webhook</span><strong>{razorpayStatus?.webhook_configured ? 'Secured' : 'Needs secret'}</strong></div>
                  <div><span>Mode</span><strong>Test Mode</strong></div>
                  <div><span>Redis / Worker</span><strong>{runtimeHealth?.redis}/{runtimeHealth?.worker}</strong></div>
                </div>
              </article>
            </div>
            <article className="panel form-panel">
              <div className="panel-head">
                <div>
                  <p className="eyebrow">DEVELOPER / TEST MODE</p>
                  <h2>Hidden demo tools</h2>
                </div>
              </div>
              <p className="muted-note">
                Merchants never see this on Overview. Use only for hackathon demos to generate a Razorpay Test Mode failure.
              </p>
              <div className="form-grid">
                <button
                  type="button"
                  className="ghost-btn"
                  onClick={() => void startTestPayment()}
                  disabled={paymentLoading || !razorpayStatus?.configured}
                >
                  {paymentLoading ? 'Opening…' : 'Generate test failure (₹2,999)'}
                </button>
              </div>
            </article>
          </section>
        )}
      </div>
    </div>
    </div>
  )
}

export default App
