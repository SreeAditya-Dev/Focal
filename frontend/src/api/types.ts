export interface DetectedIssue {
  type: 'blur' | 'underexposure' | 'overexposure' | 'noise' | 'corruption' | 'defect';
  severity: 'none' | 'low' | 'medium' | 'high';
  severity_score: number;
  confidence: number;
  rule_confidence: number;
  cnn_confidence: number;
  evidence: string[];
}

export interface AnalysisResponse {
  id?: number;
  filename: string;
  width: number;
  height: number;
  file_size?: number;
  quality_score: number;
  quality_label: 'EXCELLENT' | 'ACCEPTABLE' | 'POOR' | 'UNUSABLE';
  issues: DetectedIssue[];
  stats: Record<string, number>;
  summary: string;
  model_version: string;
  model_loaded: boolean;
  processing_time_ms: number;
  timings_ms?: Record<string, number>;
  heatmap_issue?: string | null;
  heatmap_base64?: string | null;
  uncertainty?: Array<{
    issue: string;
    mean: number;
    std: number;
    flagged: boolean;
  }>;
  created_at?: string;
}

export interface BatchAnalysisResponse {
  total: number;
  successful: number;
  failed: number;
  total_time_ms: number;
  results: AnalysisResponse[];
  errors: Array<{ filename: string; error: string }>;
}

export interface HistoryListResponse {
  total: number;
  page: number;
  page_size: number;
  items: AnalysisResponse[];
}

export interface HealthResponse {
  status: string;
  model_version: string;
  model_loaded: boolean;
  device: string;
  supported_issues: string[];
}

