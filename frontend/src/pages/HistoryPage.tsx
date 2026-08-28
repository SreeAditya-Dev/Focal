import React, { useState, useEffect } from "react";
import { api } from "../api/client";
import { HistoryItem, AnalysisResult } from "../api/types";
import { IssueCard } from "../components/IssueCard";
import { MetricsBreakdown } from "../components/MetricsBreakdown";
import {
  History,
  Search,
  Trash2,
  Eye,
  RefreshCw,
  Loader2,
  ChevronLeft,
  ChevronRight,
  AlertCircle,
  FileSpreadsheet,
} from "lucide-react";

export const HistoryPage: React.FC = () => {
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState<number>(0);
  const [page, setPage] = useState<number>(1);
  const [totalPages, setTotalPages] = useState<number>(1);
  const [filterLabel, setFilterLabel] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Selected item modal
  const [selectedRecord, setSelectedRecord] = useState<AnalysisResult | null>(null);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);

  const fetchHistory = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.getHistory(
        page,
        15,
        filterLabel === "ALL" ? undefined : filterLabel,
        searchQuery || undefined
      );
      setHistoryItems(data.items);
      setTotal(data.total);
      setTotalPages(data.total_pages);
    } catch (err: any) {
      console.error("Failed to load history:", err);
      setError("Failed to load analysis history from server.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [page, filterLabel]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchHistory();
  };

  const handleViewDetail = async (id: number) => {
    setLoadingDetail(true);
    try {
      const detail = await api.getHistoryDetail(id);
      setSelectedRecord(detail);
    } catch (err) {
      alert("Failed to load record details.");
    } finally {
      setLoadingDetail(false);
    }
  };

  const handleDelete = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm("Are you sure you want to delete this analysis record?")) return;

    try {
      await api.deleteHistoryRecord(id);
      fetchHistory();
      if (selectedRecord?.id === id) {
        setSelectedRecord(null);
      }
    } catch (err) {
      alert("Failed to delete record.");
    }
  };

  const handleClearAll = async () => {
    if (!confirm("Are you sure you want to clear all analysis history?")) return;
    try {
      await api.clearAllHistory();
      fetchHistory();
      setSelectedRecord(null);
    } catch (err) {
      alert("Failed to clear history.");
    }
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
    <div className="space-y-6 pb-16">
      {/* Top Header & Search */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 glass-panel p-5 rounded-2xl border border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <History className="h-5 w-5 text-indigo-400" />
            <h2 className="text-lg font-bold text-white">Analysis History & Audit Log</h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Persisted records of all evaluated images with quality scores and issues.
          </p>
        </div>

        {total > 0 && (
          <button
            onClick={handleClearAll}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 text-xs font-semibold transition-colors self-start sm:self-auto"
          >
            <Trash2 className="h-3.5 w-3.5" />
            <span>Clear History</span>
          </button>
        )}
      </div>

      {/* Filters & Search Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* Tier filter tabs */}
        <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
          {["ALL", "EXCELLENT", "ACCEPTABLE", "POOR", "UNUSABLE"].map((lbl) => (
            <button
              key={lbl}
              onClick={() => {
                setFilterLabel(lbl);
                setPage(1);
              }}
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

        {/* Search input */}
        <form onSubmit={handleSearchSubmit} className="flex items-center gap-2">
          <div className="relative">
            <Search className="h-3.5 w-3.5 text-slate-400 absolute left-3 top-2.5" />
            <input
              type="text"
              placeholder="Search filename..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 pr-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            />
          </div>
          <button
            type="submit"
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 font-semibold transition-colors"
          >
            Search
          </button>
        </form>
      </div>

      {/* History Table */}
      {isLoading ? (
        <div className="glass-card rounded-2xl p-12 text-center border border-slate-800">
          <Loader2 className="h-8 w-8 text-indigo-500 animate-spin mx-auto mb-3" />
          <p className="text-xs text-slate-400">Loading audit history...</p>
        </div>
      ) : historyItems.length === 0 ? (
        <div className="glass-card rounded-2xl p-12 text-center border border-slate-800">
          <History className="h-10 w-10 text-slate-600 mx-auto mb-3" />
          <h3 className="text-sm font-bold text-slate-200">No Analysis History Found</h3>
          <p className="text-xs text-slate-500 mt-1 max-w-sm mx-auto">
            Upload images from the Analyze tab to build your quality audit history.
          </p>
        </div>
      ) : (
        <div className="glass-panel rounded-2xl overflow-hidden border border-slate-800">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/90 text-slate-400 font-mono text-[11px] border-b border-slate-800 uppercase">
                <tr>
                  <th className="py-3 px-4">Date & Time</th>
                  <th className="py-3 px-4">Filename</th>
                  <th className="py-3 px-4">Resolution</th>
                  <th className="py-3 px-4">Quality Score</th>
                  <th className="py-3 px-4">Status Band</th>
                  <th className="py-3 px-4">Detected Faults</th>
                  <th className="py-3 px-4">Latency</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-sans">
                {historyItems.map((item) => (
                  <tr
                    key={item.id}
                    className="hover:bg-slate-800/30 transition-colors cursor-pointer"
                    onClick={() => handleViewDetail(item.id)}
                  >
                    <td className="py-3 px-4 font-mono text-slate-400 text-[11px]">
                      {new Date(item.created_at).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 font-medium text-slate-200 truncate max-w-[200px]">
                      {item.filename}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-400">
                      {item.width}×{item.height}
                    </td>
                    <td className="py-3 px-4 font-bold font-mono">
                      <span
                        className={
                          item.quality_score >= 70
                            ? "text-emerald-400"
                            : item.quality_score >= 40
                            ? "text-amber-400"
                            : "text-rose-400"
                        }
                      >
                        {item.quality_score.toFixed(1)}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase border ${getLabelBadge(
                          item.quality_label
                        )}`}
                      >
                        {item.quality_label}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {item.issues_summary.length === 0 ? (
                        <span className="text-slate-500 font-mono text-[11px]">
                          Clean
                        </span>
                      ) : (
                        <div className="flex flex-wrap gap-1">
                          {item.issues_summary.map((t, idx) => (
                            <span
                              key={idx}
                              className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-slate-300"
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-400 text-[11px]">
                      {item.processing_time_ms.toFixed(0)} ms
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleViewDetail(item.id);
                          }}
                          className="p-1 rounded-md hover:bg-slate-800 text-indigo-400 hover:text-indigo-300"
                          title="Inspect Detail"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        <button
                          onClick={(e) => handleDelete(item.id, e)}
                          className="p-1 rounded-md hover:bg-slate-800 text-slate-500 hover:text-rose-400"
                          title="Delete Record"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div className="px-4 py-3 bg-slate-900/60 border-t border-slate-800 flex items-center justify-between text-xs">
              <span className="text-slate-400 font-mono text-[11px]">
                Showing page {page} of {totalPages} ({total} records total)
              </span>
              <div className="flex items-center gap-1">
                <button
                  disabled={page <= 1}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:pointer-events-none text-slate-300"
                >
                  <ChevronLeft className="h-4 w-4" />
                </button>
                <button
                  disabled={page >= totalPages}
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  className="p-1 rounded bg-slate-800 hover:bg-slate-700 disabled:opacity-40 disabled:pointer-events-none text-slate-300"
                >
                  <ChevronRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Detail Modal */}
      {selectedRecord && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
          <div className="glass-panel w-full max-w-4xl rounded-2xl border border-slate-700 max-h-[90vh] overflow-y-auto p-6 space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-base font-bold text-slate-100">
                  {selectedRecord.filename}
                </h3>
                <span className="text-xs font-mono text-slate-400">
                  {selectedRecord.width} × {selectedRecord.height} px • Score:{" "}
                  <span className="font-bold text-indigo-400">
                    {selectedRecord.quality_score.toFixed(1)}
                  </span>{" "}
                  ({selectedRecord.quality_label})
                </span>
              </div>
              <button
                onClick={() => setSelectedRecord(null)}
                className="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-300"
              >
                Close
              </button>
            </div>

            <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300">
              <span className="font-bold text-indigo-400 block mb-1">Executive Summary:</span>
              {selectedRecord.summary}
            </div>

            {/* Detected issues */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                Detected Quality Degradations ({selectedRecord.issues.length})
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {selectedRecord.issues.map((issue, idx) => (
                  <IssueCard key={idx} issue={issue} />
                ))}
              </div>
            </div>

            {/* Full 47 features */}
            <MetricsBreakdown stats={selectedRecord.stats} />
          </div>
        </div>
      )}
    </div>
  );
};

