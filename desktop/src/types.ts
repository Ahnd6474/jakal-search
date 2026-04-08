export type FeedbackChoice = "A" | "B" | "tie";

export interface SourceCount {
  source: string;
  count: number;
}

export interface TopicSummary {
  label: string;
  query: string;
  score: number;
  trust: number;
  support: number;
}

export interface DocumentSummary {
  title: string;
  source: string;
  url: string;
  snippet: string;
  trust_score: number;
}

export interface SearchSummary {
  topic_count: number;
  document_count: number;
  avg_trust_score: number;
  expanded_node_count: number;
  quality_score: number;
  top_sources: SourceCount[];
  top_topics: TopicSummary[];
  top_documents: DocumentSummary[];
}

export interface ComparisonOption {
  slot: "A" | "B";
  summary: SearchSummary;
  report: string;
  tree: Record<string, unknown>;
}

export interface ComparisonResponse {
  run_id: string;
  query: string;
  created_at: string;
  options: ComparisonOption[];
  policy: {
    feedback_count: number;
    run_count: number;
    status_text: string;
  };
}

export interface FeedbackReceipt {
  run_id: string;
  choice: FeedbackChoice;
  feedback_count: number;
  message: string;
}

export interface DashboardEntry {
  action_id: string;
  title: string;
  repeat_count: number;
  comparisons: number;
  wins: number;
  losses: number;
  ties: number;
  rating: number;
  avg_quality: number;
  quality_observations: number;
  win_rate: number;
  last_selected_at: string | null;
  last_feedback_at: string | null;
}

export interface DashboardEvent {
  event_id: string;
  run_id: string;
  query: string;
  submitted_at: string;
  choice: FeedbackChoice;
}

export interface DashboardData {
  feedback_count: number;
  run_count: number;
  leaderboard: DashboardEntry[];
  recent_events: DashboardEvent[];
}
