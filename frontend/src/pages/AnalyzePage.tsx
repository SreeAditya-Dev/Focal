import React, { useState } from "react";
import { api } from "../api/client";
import { AnalysisResult, BatchAnalysisResult } from "../api/types";
import { DropZone } from "../components/DropZone";
import { ScoreGauge } from "../components/ScoreGauge";
import { IssueCard } from "../components/IssueCard";
import { HeatmapViewer } from "../components/HeatmapViewer";
import { MetricsBreakdown } from "../components/MetricsBreakdown";
import { FeatureRadar } from "../components/FeatureRadar";
import { BatchResultsTable } from "../components/BatchResultsTable";
import {
  Loader2,
  RefreshCw,
  AlertCircle,
  FileDown,
  Sparkles,
  Info,
  CheckCircle2,
} from "lucide-react";

export const AnalyzePage: React.FC = () => {
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Single analysis state
  const [singleResult, setSingleResult] = useState<AnalysisResult | null>(null);
  const [uploadedPreview, setUploadedPreview] = useState<string | null>(null);

  // Batch analysis state
  const [batchResult, setBatchResult] = useState<BatchAnalysisResult | null>(null);
  const [selectedBatchItem, setSelectedBatchItem] = useState<AnalysisResult | null>(null);

  const handleSelectSingle = async (file: File) => {
    setIsProcessing(true);
    setError(null);
    const localUrl = URL.createObjectURL(file);
    setUploadedPreview(localUrl);

    try {
      const result = await api.analyzeImage(file, true, true, true);
      setSingleResult(result);
    } catch (err: any) {
      console.error("Analysis error:", err);
      const detail = err.response?.data?.detail || err.message || "Failed to analyze image.";
      setError(detail);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSelectBatch = async (files: File[]) => {
    setIsProcessing(true);
    setError(null);

    try {
      const result = await api.analyzeBatch(files, false, true);
      setBatchResult(result);
      if (result.results.length > 0) {
        setSelectedBatchItem(result.results[0]);
      }
    } catch (err: any) {
      console.error("Batch analysis error:", err);
      const detail = err.response?.data?.detail || err.message || "Failed to process batch images.";
      setError(detail);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleExportJSON = () => {
    const dataToExport = singleResult || batchResult;
    if (!dataToExport) return;
    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `focal_analysis_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const resetAnalysis = () => {
    setSingleResult(null);
    setBatchResult(null);
    setSelectedBatchItem(null);
    setUploadedPreview(null);
    setError(null);
  };

  return (
    <div className="space-y-8 pb-16">
      {/* Hero Welcome Banner */}
      {!singleResult && !batchResult && !isProcessing && (
        <div className="text-center max-w-2xl mx-auto pt-6 space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-semibold">
            <Sparkles className="h-3.5 w-3.5" />
            <span>Hybrid Classical-CV + MobileNetV3 AI Engine</span>
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            Comprehensive Image Quality & Defect Detection
          </h1>
          <p className="text-sm text-slate-400 leading-relaxed">
            Upload any image to instantly measure blur, exposure health, noise floor, compression artifacts, and localized defects with explainable Grad-CAM heatmaps.
          </p>
        </div>
      )}

      {/* Upload Zone */}
      {!singleResult && !batchResult && (
        <div className="max-w-3xl mx-auto">
          <DropZone
            onSelectSingle={handleSelectSingle}
            onSelectBatch={handleSelectBatch}
            isProcessing={isProcessing}
            mode={mode}
            onModeChange={(m) => {
              setMode(m);
              setError(null);
            }}
          />
        </div>
      )}

      {/* Loading State */}
      {isProcessing && (
        <div className="glass-card rounded-2xl p-12 text-center max-w-md mx-auto my-8 border border-slate-800">
          <Loader2 className="h-10 w-10 text-indigo-500 animate-spin mx-auto mb-4" />
          <h3 className="text-base font-bold text-slate-100 mb-1">
            Running Hybrid Quality Pipeline...
          </h3>
          <p className="text-xs text-slate-400 font-mono">
            Extracting 47 classical spatial features & running CNN forward pass
          </p>
        </div>
      )}

      {/* Error Alert */}
      {error && (
        <div className="max-w-2xl mx-auto rounded-xl p-4 bg-rose-500/10 border border-rose-500/30 text-rose-300 flex items-start gap-3 text-xs">
          <AlertCircle className="h-5 w-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">
            <h4 className="font-bold text-sm text-rose-200">Analysis Error</h4>
            <p className="mt-1">{error}</p>
          </div>
          <button
            onClick={() => setError(null)}
            className="text-rose-400 hover:text-rose-200 font-bold text-xs"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Single Image Results Dashboard */}
      {singleResult && uploadedPreview && !isProcessing && (
        <div className="space-y-6">
          {/* Top Actions Bar */}
          <div className="flex flex-wrap items-center justify-between gap-4 glass-panel p-4 rounded-2xl border border-slate-800">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400">Analysis for:</span>
                <span className="font-bold text-sm text-slate-100">{singleResult.filename}</span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  {singleResult.width} × {singleResult.height} px ({((singleResult.file_size || 0)/1024).toFixed(1)} KB)
                </span>
              </div>
              <p className="text-xs text-indigo-300/90 mt-1 font-medium">
                {singleResult.summary}
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handleExportJSON}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-200 transition-colors shadow-sm"
              >
                <FileDown className="h-3.5 w-3.5 text-indigo-400" />
                <span>Export JSON</span>
              </button>

              <button
                onClick={resetAnalysis}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all shadow-md shadow-indigo-600/30"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                <span>New Analysis</span>
              </button>
            </div>
          </div>

          {/* Core Analytics Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Column: Score Gauge & Radar */}
            <div className="lg:col-span-4 space-y-6">
              <ScoreGauge
                score={singleResult.quality_score}
                label={singleResult.quality_label}
                processingTimeMs={singleResult.processing_time_ms}
                uncertainty={singleResult.uncertainty}
              />
              <FeatureRadar
                stats={singleResult.stats}
                qualityScore={singleResult.quality_score}
              />
            </div>

            {/* Right Column: Grad-CAM Explainability & Issues */}
            <div className="lg:col-span-8 space-y-6">
              <HeatmapViewer
                originalImage={uploadedPreview}
                heatmapBase64={singleResult.heatmap_base64}
                heatmapIssue={singleResult.heatmap_issue}
              />

              {/* Detected Issues Cards */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                    Detected Quality Degradations ({singleResult.issues.length})
                  </h3>
                  {singleResult.issues.length === 0 && (
                    <span className="inline-flex items-center gap-1 text-xs text-emerald-400 font-semibold">
                      <CheckCircle2 className="h-3.5 w-3.5" /> Image meets pristine standards
                    </span>
                  )}
                </div>

                {singleResult.issues.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {singleResult.issues.map((issue, idx) => (
                      <IssueCard key={idx} issue={issue} />
                    ))}
                  </div>
                ) : (
                  <div className="glass-card rounded-xl p-6 text-center border border-emerald-500/20 bg-emerald-500/5">
                    <CheckCircle2 className="h-8 w-8 text-emerald-400 mx-auto mb-2" />
                    <h4 className="text-sm font-bold text-slate-100">No Defects Detected</h4>
                    <p className="text-xs text-slate-400 mt-1 max-w-sm mx-auto">
                      All physical measurements (Laplacian sharpness, noise floor, exposure clipping, blockiness) are within clean reference thresholds.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Full 47 Feature Inspection Table */}
          <MetricsBreakdown stats={singleResult.stats} />
        </div>
      )}

      {/* Batch Results View */}
      {batchResult && !isProcessing && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-white">Batch Processing Completed</h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Processed {batchResult.successful} of {batchResult.total} images in {(batchResult.total_processing_time_ms/1000).toFixed(2)}s
              </p>
            </div>
            <button
              onClick={resetAnalysis}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all shadow-md shadow-indigo-600/30"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              <span>New Upload</span>
            </button>
          </div>

          <BatchResultsTable
            batchData={batchResult}
            onSelectResult={(item) => setSelectedBatchItem(item)}
          />

          {/* Modal / Side Detail View for Selected Batch Item */}
          {selectedBatchItem && (
            <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-6">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="text-sm font-bold text-slate-100">
                  Item Detail: {selectedBatchItem.filename}
                </h3>
                <span className="text-xs font-mono text-indigo-400">
                  Score: {selectedBatchItem.quality_score.toFixed(1)} / 100 ({selectedBatchItem.quality_label})
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {selectedBatchItem.issues.map((issue, idx) => (
                  <IssueCard key={idx} issue={issue} />
                ))}
              </div>

              <MetricsBreakdown stats={selectedBatchItem.stats} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

