import { invoke } from "@tauri-apps/api/core";
import type {
  ComparisonResponse,
  DashboardData,
  FeedbackChoice,
  FeedbackReceipt,
} from "./types";

export async function runComparison(query: string): Promise<ComparisonResponse> {
  return invoke<ComparisonResponse>("run_comparison", {
    payload: { query },
  });
}

export async function submitFeedback(
  runId: string,
  choice: FeedbackChoice,
): Promise<FeedbackReceipt> {
  return invoke<FeedbackReceipt>("submit_feedback", {
    payload: { runId, choice },
  });
}

export async function getDashboard(): Promise<DashboardData> {
  return invoke<DashboardData>("get_dashboard");
}
