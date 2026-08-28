import axios from "axios";
import {
  AnalysisResult,
  BatchAnalysisResult,
  HealthStatus,
  HistoryListResponse,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/v1";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Accept": "application/json",
  },
});

export const api = {
  /**
   * Analyze a single image.
   */
  async analyzeImage(
    file: File,
    includeHeatmap: boolean = true,
    uncertainty: boolean = true,
    saveRecord: boolean = true
  ): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await apiClient.post<AnalysisResult>("/analyze", formData, {
      params: {
        include_heatmap: includeHeatmap,
        uncertainty: uncertainty,
        save_record: saveRecord,
      },
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  },

  /**
   * Analyze multiple images in batch.
   */
  async analyzeBatch(
    files: File[],
    includeHeatmap: boolean = false,
    saveRecords: boolean = true
  ): Promise<BatchAnalysisResult> {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });

    const response = await apiClient.post<BatchAnalysisResult>(
      "/analyze/batch",
      formData,
      {
        params: {
          include_heatmap: includeHeatmap,
          save_records: saveRecords,
        },
        headers: {
          "Content-Type": "multipart/form-data",
        },
      }
    );

    return response.data;
  },

  /**
   * Fetch paginated history list.
   */
  async getHistory(
    page: number = 1,
    limit: number = 20,
    qualityLabel?: string,
    search?: string
  ): Promise<HistoryListResponse> {
    const response = await apiClient.get<HistoryListResponse>("/history", {
      params: {
        page,
        limit,
        quality_label: qualityLabel,
        search,
      },
    });
    return response.data;
  },

  /**
   * Fetch previous analysis by ID.
   */
  async getHistoryDetail(id: number): Promise<AnalysisResult> {
    const response = await apiClient.get<AnalysisResult>(`/history/${id}`);
    return response.data;
  },

  /**
   * Delete an analysis entry.
   */
  async deleteHistoryRecord(id: number): Promise<void> {
    await apiClient.delete(`/history/${id}`);
  },

  /**
   * Clear all history entries.
   */
  async clearAllHistory(): Promise<void> {
    await apiClient.delete("/history");
  },

  /**
   * Check backend health and model status.
   */
  async getHealth(): Promise<HealthStatus> {
    const response = await apiClient.get<HealthStatus>("/health");
    return response.data;
  },
};

