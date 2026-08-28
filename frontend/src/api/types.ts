export type QualityLabel = "EXCELLENT" | "ACCEPTABLE" | "POOR" | "UNUSABLE";

export interface Issue {
  type: string;
  severity: "low" | "medium" | "high" | "severe" | string;
  severity_score: number;
  confidence: number;
  rule_confidence: number;
  cnn_confidence: number;
  evidence: string[];
}

export interface AnalysisResult {
  id?: number;
  filename: string;
  width: number;
  height: number;
  file_size: number;
  quality_score: number;
  quality_label: QualityLabel;
  issues: Issue[];
  stats: Record<string, number>;
  summary: string;
  model_version: string;
  model_loaded: boolean;
  processing_time_ms: number;
  timings_ms: Record<string, number>;
  uncertainty?: number | null;
  heatmap_base64?: string | null;
  heatmap_issue?: string | null;
  created_at?: string;
  previewUrl?: string; // Client-side object URL for display
}

export interface BatchAnalysisResult {
  total: number;
  successful: number;
  failed: number;
  results: AnalysisResult[];
  total_processing_time_ms: number;
}

export interface HistoryItem {
  id: number;
  filename: string;
  file_size: number;
  width: number;
  height: number;
  quality_score: number;
  quality_label: QualityLabel;
  issue_count: number;
  issues_summary: string[];
  processing_time_ms: number;
  created_at: string;
}

export interface HistoryListResponse {
  items: HistoryItem[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
}

export interface HealthStatus {
  status: string;
  version: string;
  model_version: string;
  model_loaded: boolean;
  device: string;
  database: string;
  timestamp: string;
}

