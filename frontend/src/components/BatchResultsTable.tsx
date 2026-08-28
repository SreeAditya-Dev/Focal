import React, { useState } from "react";
import { BatchAnalysisResult, AnalysisResult } from "../api/types";
import { Download, CheckCircle2, AlertTriangle, XCircle, Award, Eye, FileSpreadsheet } from "lucide-react";

interface BatchResultsTableProps {
  batchData: BatchAnalysisResult;
  onSelectResult: (result: AnalysisResult) => void;
}

export const BatchResultsTable: React.FC<BatchResultsTableProps> = ({
  batchData,
  onSelectResult,
}) => {
  const [filterLabel, setFilterLabel] = useState<string>("ALL");

  const results = batchData.results || [];
  const avgScore =
    results.length > 0
      ? results.reduce((acc, r) => acc + r.quality_score, 0) / results.length
      : 0;

  const passedCount = results.filter(
    (r) => r.quality_label === "EXCELLENT" || r.quality_label === "ACCEPTABLE"
  ).length;
  const passRate = results.length > 0 ? (passedCount / results.length) * 100 : 0;

  const filteredResults =
    filterLabel === "ALL"
      ? results
      : results.filter((r) => r.quality_label === filterLabel);

  const exportCSV = () => {
    const headers = [
      "Filename",
      "Width",
      "Height",
      "Quality Score",
      "Quality Label",
      "Issues Count",
      "Detected Issues",
      "Latency (ms)",
    ];

    const rows = results.map((r) => [
      r.filename,
      r.width,
      r.height,
      r.quality_score.toFixed(1),
      r.quality_label,
      r.issues.length,
      r.issues.map((i) => `${i.type}(${i.severity})`).join("; "),
      r.processing_time_ms.toFixed(1),
    ]);

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `focal_batch_analysis_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getLabelBadge = (label: string) => {
    switch (label) {
      case "EXCELLENT":
        return "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";
      case "ACCEPTABLE":
        return "bg-blue-500/10 text-blue-400 border-blue-500/30";
      case "POOR":
        return "bg-amber-500/10 text-amber-400 border-amber-500/30";
      case "UNUSABLE":
        return "bg-rose-500/10 text-rose-400 border-rose-500/30";
      default:
        return "bg-slate-800 text-slate-400 border-slate-700";
    }
  };

  return (
    <div className="space-y-6">
      {/* Batch KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="glass-card rounded-xl p-4 border border-slate-800">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Total Processed
          </span>
          <div className="text-2xl font-bold text-slate-100 mt-1">
            {batchData.successful} <span className="text-xs text-slate-500 font-normal">/ {batchData.total} files</span>
          </div>
        </div>

        <div className="glass-card rounded-xl p-4 border border-slate-800">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Mean Quality Score
          </span>
          <div className="text-2xl font-bold text-indigo-400 mt-1">
            {avgScore.toFixed(1)} <span className="text-xs text-slate-500 font-normal">/ 100</span>
          </div>
        </div>

        <div className="glass-card rounded-xl p-4 border border-slate-800">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Acceptance Rate
          </span>
          <div className="text-2xl font-bold text-emerald-400 mt-1">
            {passRate.toFixed(1)}%
          </div>
        </div>

        <div className="glass-card rounded-xl p-4 border border-slate-800">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Batch Throughput
          </span>
          <div className="text-2xl font-bold text-slate-200 mt-1">
            {(batchData.total_processing_time_ms / 1000).toFixed(2)}s
          </div>
        </div>
      </div>

      {/* Table Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
          {["ALL", "EXCELLENT", "ACCEPTABLE", "POOR", "UNUSABLE"].map((lbl) => (
            <button
              key={lbl}
              onClick={() => setFilterLabel(lbl)}
              className={`px-3 py-1 rounded font-medium transition-all ${
                filterLabel === lbl
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {lbl}
            </button>
          ))}
        </div>

        <button
          onClick={exportCSV}
          className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-semibold text-slate-200 transition-colors shadow-sm"
        >
          <FileSpreadsheet className="h-3.5 w-3.5 text-emerald-400" />
          <span>Export CSV Report</span>
        </button>
      </div>

      {/* Results Table */}
      <div className="glass-panel rounded-2xl overflow-hidden border border-slate-800">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-900/90 text-slate-400 font-mono text-[11px] border-b border-slate-800 uppercase">
              <tr>
                <th className="py-3 px-4">Filename</th>
                <th className="py-3 px-4">Resolution</th>
                <th className="py-3 px-4">Quality Score</th>
                <th className="py-3 px-4">Tier Band</th>
                <th className="py-3 px-4">Detected Faults</th>
                <th className="py-3 px-4">Latency</th>
                <th className="py-3 px-4 text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-sans">
              {filteredResults.map((result, idx) => (
                <tr
                  key={idx}
                  className="hover:bg-slate-800/30 transition-colors cursor-pointer"
                  onClick={() => onSelectResult(result)}
                >
                  <td className="py-3 px-4 font-medium text-slate-200 truncate max-w-[200px]">
                    {result.filename}
                  </td>
                  <td className="py-3 px-4 font-mono text-slate-400">
                    {result.width}×{result.height}
                  </td>
                  <td className="py-3 px-4 font-bold font-mono">
                    <span
                      className={
                        result.quality_score >= 70
                          ? "text-emerald-400"
                          : result.quality_score >= 40
                          ? "text-amber-400"
                          : "text-rose-400"
                      }
                    >
                      {result.quality_score.toFixed(1)}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase border ${getLabelBadge(
                        result.quality_label
                      )}`}
                    >
                      {result.quality_label}
                    </span>
                  </td>
                  <td className="py-3 px-4">
                    {result.issues.length === 0 ? (
                      <span className="text-slate-500 font-mono text-[11px]">
                        Clean
                      </span>
                    ) : (
                      <div className="flex flex-wrap gap-1">
                        {result.issues.map((i, iIdx) => (
                          <span
                            key={iIdx}
                            className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-slate-300"
                          >
                            {i.type} ({i.severity})
                          </span>
                        ))}
                      </div>
                    )}
                  </td>
                  <td className="py-3 px-4 font-mono text-slate-400 text-[11px]">
                    {result.processing_time_ms.toFixed(0)} ms
                  </td>
                  <td className="py-3 px-4 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectResult(result);
                      }}
                      className="p-1 rounded-md hover:bg-slate-800 text-indigo-400 hover:text-indigo-300"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

