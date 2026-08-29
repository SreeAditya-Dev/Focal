import axios from 'axios';
import { AnalysisResponse, BatchAnalysisResponse, HistoryListResponse, HealthResponse } from './types';

const api = axios.create({
  baseURL: '/api/v1',
  headers: {
    'Accept': 'application/json',
  },
});

export const checkHealth = async (): Promise<HealthResponse> => {
  const res = await api.get<HealthResponse>('/health');
  return res.data;
};

export const analyzeImage = async (
  file: File,
  includeHeatmap = true,
  uncertainty = true
): Promise<AnalysisResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post<AnalysisResponse>('/analyze', formData, {
    params: {
      include_heatmap: includeHeatmap,
      uncertainty: uncertainty,
    },
  });
  return res.data;
};

export const analyzeBatch = async (
  files: File[],
  includeHeatmap = false
): Promise<BatchAnalysisResponse> => {
  const formData = new FormData();
  files.forEach((f) => formData.append('files', f));
  const res = await api.post<BatchAnalysisResponse>('/analyze/batch', formData, {
    params: {
      include_heatmap: includeHeatmap,
    },
  });
  return res.data;
};

export const getHistory = async (
  page = 1,
  pageSize = 20,
  labelFilter?: string
  limit = 20,
  qualityLabel?: string,
  search?: string
): Promise<HistoryListResponse> => {
  const res = await api.get<HistoryListResponse>('/history', {
    params: {
      page,
      page_size: pageSize,
      label_filter: labelFilter,
      limit,
      quality_label: qualityLabel || undefined,
      search: search || undefined,
    },
  });
  return res.data;
};

export const getHistoryItem = async (id: number): Promise<AnalysisResponse> => {
  const res = await api.get<AnalysisResponse>(`/history/${id}`);
  return res.data;
};

export const deleteHistoryItem = async (id: number): Promise<void> => {
  await api.delete(`/history/${id}`);
};

export const clearHistory = async (): Promise<void> => {
  await api.delete('/history');
};

