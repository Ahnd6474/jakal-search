import { FormEvent, useEffect, useState } from "react";
import { getDashboard, runComparison, submitFeedback } from "./api";
import type {
  ComparisonOption,
  ComparisonResponse,
  DashboardData,
  FeedbackChoice,
} from "./types";

function App() {
  const [query, setQuery] = useState("graph search systems");
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [feedbackNotice, setFeedbackNotice] = useState<string | null>(null);
  const [lastChoice, setLastChoice] = useState<FeedbackChoice | null>(null);
  const [showOps, setShowOps] = useState(false);

  useEffect(() => {
    void refreshDashboard();
  }, []);

  async function refreshDashboard() {
    try {
      const nextDashboard = await getDashboard();
      setDashboard(nextDashboard);
    } catch (nextError) {
      const message = nextError instanceof Error ? nextError.message : String(nextError);
      setError(message);
    }
  }

  async function generateComparison() {
    setError(null);
    setFeedbackNotice(null);
    setLastChoice(null);
    setIsLoading(true);
    try {
      const result = await runComparison(query);
      setComparison(result);
      await refreshDashboard();
    } catch (nextError) {
      const message = nextError instanceof Error ? nextError.message : String(nextError);
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleGenerate(event: FormEvent) {
    event.preventDefault();
    await generateComparison();
  }

  async function handleFeedback(choice: FeedbackChoice) {
    if (!comparison) {
      return;
    }
    setIsSubmitting(true);
    setError(null);
    try {
      const receipt = await submitFeedback(comparison.run_id, choice);
      setFeedbackNotice(receipt.message);
      setLastChoice(choice);
      await refreshDashboard();
    } catch (nextError) {
      const message = nextError instanceof Error ? nextError.message : String(nextError);
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Pairwise Search Trainer</p>
          <h1>설정 없이, 결과만 비교하는 jakal-search</h1>
          <p className="hero-text">
            사용자는 검색어를 넣고 어떤 결과가 더 좋은지만 고릅니다. 내부에서는 숨겨진 탐색 정책이
            자동으로 서로 경쟁하고, 피드백을 받아 다음 비교를 조정합니다.
          </p>
        </div>
        <div className="hero-status">
          <div className="status-chip">Hidden policies active</div>
          <div className="status-chip alt">
            Feedback{" "}
            <strong>{dashboard?.feedback_count ?? comparison?.policy.feedback_count ?? 0}</strong>
          </div>
        </div>
      </section>

      <section className="query-card">
        <form className="query-form" onSubmit={handleGenerate}>
          <label className="query-label" htmlFor="query">
            비교할 질문
          </label>
          <div className="query-row">
            <input
              id="query"
              className="query-input"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="예: graph search systems"
            />
            <button className="primary-button" disabled={isLoading || !query.trim()} type="submit">
              {isLoading ? "비교 생성 중..." : "비교 시작"}
            </button>
          </div>
        </form>
        <p className="query-caption">
          노출되는 건 결과 A/B뿐입니다. 깊이, 반복 횟수, trust 기준 같은 파라미터는 숨겨져 있습니다.
        </p>
      </section>

      {error ? <section className="message error">{error}</section> : null}
      {feedbackNotice ? <section className="message success">{feedbackNotice}</section> : null}

      {comparison ? (
        <>
          <section className="compare-header">
            <div>
              <p className="eyebrow">Current Match</p>
              <h2>{comparison.query}</h2>
            </div>
            <div className="policy-note">
              <span>{comparison.policy.status_text}</span>
              <span>Run #{comparison.policy.run_count}</span>
            </div>
          </section>

          <section className="comparison-grid">
            {comparison.options.map((option) => (
              <ComparisonCard key={option.slot} option={option} />
            ))}
          </section>

          <section className="feedback-bar">
            <button
              className={`feedback-button ${lastChoice === "A" ? "selected" : ""}`}
              disabled={isSubmitting || lastChoice !== null}
              onClick={() => void handleFeedback("A")}
              type="button"
            >
              A가 더 좋음
            </button>
            <button
              className={`feedback-button neutral ${lastChoice === "tie" ? "selected" : ""}`}
              disabled={isSubmitting || lastChoice !== null}
              onClick={() => void handleFeedback("tie")}
              type="button"
            >
              비슷함
            </button>
            <button
              className={`feedback-button ${lastChoice === "B" ? "selected" : ""}`}
              disabled={isSubmitting || lastChoice !== null}
              onClick={() => void handleFeedback("B")}
              type="button"
            >
              B가 더 좋음
            </button>
            <button
              className="secondary-button"
              disabled={isLoading}
              onClick={() => void generateComparison()}
              type="button"
            >
              다음 비교 생성
            </button>
          </section>
        </>
      ) : (
        <section className="empty-card">
          <p>비교를 시작하면 A/B 결과가 나란히 나타납니다.</p>
        </section>
      )}

      <section className="ops-panel">
        <button className="ops-toggle" onClick={() => setShowOps((value) => !value)} type="button">
          {showOps ? "운영 진단 닫기" : "운영 진단 보기"}
        </button>
        {showOps && dashboard ? <DashboardPanel dashboard={dashboard} /> : null}
      </section>
    </main>
  );
}

function ComparisonCard({ option }: { option: ComparisonOption }) {
  return (
    <article className="result-card">
      <header className="result-header">
        <div className="slot-badge">{option.slot}</div>
        <div>
          <h3>Result {option.slot}</h3>
          <p className="result-kicker">사용자에게는 결과만 보입니다.</p>
        </div>
      </header>

      <section className="metrics-grid">
        <Metric label="문서" value={option.summary.document_count} />
        <Metric label="토픽" value={option.summary.topic_count} />
        <Metric label="평균 신뢰" value={option.summary.avg_trust_score.toFixed(2)} />
        <Metric label="내부 품질" value={option.summary.quality_score.toFixed(2)} />
      </section>

      <section className="content-block">
        <div className="section-title">Top Topics</div>
        <div className="chip-list">
          {option.summary.top_topics.length > 0 ? (
            option.summary.top_topics.map((topic) => (
              <span className="topic-chip" key={`${option.slot}-${topic.query}`}>
                {topic.label}
              </span>
            ))
          ) : (
            <span className="topic-chip muted">하위 토픽이 아직 없습니다</span>
          )}
        </div>
      </section>

      <section className="content-block">
        <div className="section-title">Top Documents</div>
        <div className="doc-list">
          {option.summary.top_documents.map((doc) => (
            <div className="doc-item" key={`${option.slot}-${doc.url}`}>
              <strong>{doc.title}</strong>
              <span>
                {doc.source} · trust {doc.trust_score.toFixed(2)}
              </span>
              <p>{doc.snippet}</p>
              <small>{doc.url}</small>
            </div>
          ))}
        </div>
      </section>

      <details className="report-box">
        <summary>리포트 보기</summary>
        <pre>{option.report}</pre>
      </details>
    </article>
  );
}

function DashboardPanel({ dashboard }: { dashboard: DashboardData }) {
  return (
    <div className="dashboard-grid">
      <section className="ops-card">
        <div className="section-title">Policy Leaderboard</div>
        <div className="leaderboard-list">
          {dashboard.leaderboard.map((entry) => (
            <div className="leaderboard-row" key={entry.action_id}>
              <div>
                <strong>{entry.title}</strong>
                <p>
                  {entry.action_id} · repeat {entry.repeat_count}
                </p>
              </div>
              <div className="leaderboard-stats">
                <span>rating {entry.rating.toFixed(0)}</span>
                <span>win {Math.round(entry.win_rate * 100)}%</span>
                <span>cmp {entry.comparisons}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="ops-card">
        <div className="section-title">Recent Feedback</div>
        <div className="event-list">
          {dashboard.recent_events.length > 0 ? (
            dashboard.recent_events.map((event) => (
              <div className="event-row" key={event.event_id}>
                <strong>{event.choice}</strong>
                <span>{event.query}</span>
                <span>{event.submitted_at}</span>
              </div>
            ))
          ) : (
            <p className="muted-line">아직 피드백이 없습니다.</p>
          )}
        </div>
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
