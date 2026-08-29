import React, { useState } from 'react';
import { DropZone } from '../components/DropZone';
import { ScoreGauge } from '../components/ScoreGauge';
import { IssueCard } from '../components/IssueCard';
import { HeatmapViewer } from '../components/HeatmapViewer';
import { FeatureRadar } from '../components/FeatureRadar';
import { MetricsBreakdown } from '../components/MetricsBreakdown';
import { BatchResultsTable } from '../components/BatchResultsTable';
import { analyzeImage, analyzeBatch } from '../api/client';
import { AnalysisResponse, BatchAnalysisResponse } from '../api/types';
import { AlertCircle, Download, RefreshCw, FileText } from 'lucide-react';

export const AnalyzePage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [batchResult, setBatchResult] = useState<BatchAnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [batchMode, setBatchMode] = useState(false);

  const handleSingleFile = async (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setIsLoading(true);
    setErrorMessage(null);
    setBatchResult(null);

    try {
      const res = await analyzeImage(file, true, true);
      setAnalysisResult(res);
    } catch (err: any) {
      setErrorMessage(err?.response?.data?.detail || 'Analysis request failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBatchFiles = async (files: File[]) => {
    setIsLoading(true);
    setErrorMessage(null);
    setAnalysisResult(null);
    setPreviewUrl(null);

    try {
      const res = await analyzeBatch(files, false);
      setBatchResult(res);
    } catch (err: any) {
      setErrorMessage(err?.response?.data?.detail || 'Batch analysis request failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const exportJSON = () => {
    if (!analysisResult) return;
    const blob = new Blob([JSON.stringify(analysisResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `focal_analysis_${analysisResult.filename}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="space-y-6">
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-300 flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span className="text-sm font-medium">{errorMessage}</span>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 space-y-6">
          <DropZone
            onFileSelected={handleSingleFile}
            onBatchSelected={handleBatchFiles}
            isLoading={isLoading}
            batchMode={batchMode}
            setBatchMode={setBatchMode}
          />

          {isLoading && (
            <div className="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center text-center">
              <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin mb-3" />
              <h4 className="text-sm font-bold text-slate-100">Extracting 47 Features & Running CNN</h4>
              <p className="text-xs text-slate-400 mt-1">Calibrating Bayesian probabilities...</p>
            </div>
          )}

          {analysisResult && !batchResult && (
            <>
              <ScoreGauge
                score={analysisResult.quality_score}
                label={analysisResult.quality_label}
                processingTimeMs={analysisResult.processing_time_ms}
              />
              <FeatureRadar stats={analysisResult.stats} />
            </>
          )}
        </div>

        <div className="lg:col-span-7 space-y-6">
          {batchResult && (
            <BatchResultsTable
              batchData={batchResult}
              onSelectResult={(r) => {
                setAnalysisResult(r);
                setBatchResult(null);
                setBatchMode(false);
              }}
            />
          )}

          {analysisResult && !batchResult && (
            <>
              <div className="glass-panel rounded-2xl p-5">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <h3 className="text-base font-bold text-slate-100">{analysisResult.filename}</h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      {analysisResult.width} × {analysisResult.height} px •{' '}
                      {analysisResult.summary}
                    </p>
                  </div>
                  <button
                    onClick={exportJSON}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-semibold text-slate-300 transition-colors"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    Export JSON
                  </button>
                </div>

                <div className="mt-4 space-y-3">
                  <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                    Detected Quality Degradations ({analysisResult.issues.length})
                  </span>
                  {analysisResult.issues.length === 0 ? (
                    <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-sm font-medium">
                      ✓ Flawless image: No visual defects or degradations detected.
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 gap-3">
                      {analysisResult.issues.map((issue, idx) => (
                        <IssueCard key={idx} issue={issue} />
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <HeatmapViewer
                originalImage={previewUrl}
                heatmapBase64={analysisResult.heatmap_base64}
                heatmapIssue={analysisResult.heatmap_issue}
              />

              <MetricsBreakdown stats={analysisResult.stats} />
            </>
          )}
        </div>
      </div>
    </div>
  );
};

