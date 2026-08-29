import React, { useState, useEffect } from 'react';
import { getHistory, getHistoryItem, deleteHistoryItem, clearHistory } from '../api/client';
import { HistoryItem, AnalysisResponse } from '../api/types';
import { Trash2, Search, Filter, RefreshCw, X, FileText } from 'lucide-react';
import { ScoreGauge } from '../components/ScoreGauge';
import { IssueCard } from '../components/IssueCard';
import { MetricsBreakdown } from '../components/MetricsBreakdown';

export const HistoryPage: React.FC = () => {
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [labelFilter, setLabelFilter] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<AnalysisResponse | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const res = await getHistory(page, 20, labelFilter || undefined, searchTerm || undefined);
      setHistoryItems(res.items || []);
      setTotal(res.total || 0);
    } catch (err) {
      console.error('Failed to load history', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [page, labelFilter]);

  const handleSelectRecord = async (item: HistoryItem) => {
    setIsLoadingDetail(true);
    try {
      const detail = await getHistoryItem(item.id);
      setSelectedRecord(detail);
    } catch (err) {
      console.error('Failed to fetch detail', err);
    } finally {
      setIsLoadingDetail(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Delete this analysis record?')) return;
    try {
      await deleteHistoryItem(id);
      if (selectedRecord?.id === id) setSelectedRecord(null);
      fetchHistory();
    } catch (err) {
      console.error(err);
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm('Are you sure you want to clear all history records?')) return;
    try {
      await clearHistory();
      setSelectedRecord(null);
      fetchHistory();
    } catch (err) {
      console.error(err);
    }
  };

  const filteredItems = (historyItems || []).filter((item) =>
    (item.filename || '').toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-foreground">Audit History Log</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Persisted inspection logs stored in database ({total} total analyses)
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={fetchHistory}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-card hover:bg-muted border border-border text-xs font-semibold text-popover-foreground transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          {total > 0 && (
            <button
              onClick={handleClearAll}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 border border-rose-500/20 text-xs font-semibold text-rose-300 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear History
            </button>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 min-w-[240px]">
          <Search className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by filename..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && fetchHistory()}
            className="w-full bg-card/80 border border-border rounded-xl pl-9 pr-4 py-2 text-xs text-card-foreground placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors"
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <select
            value={labelFilter}
            onChange={(e) => {
              setLabelFilter(e.target.value);
              setPage(1);
            }}
            className="bg-card/80 border border-border rounded-xl px-3 py-2 text-xs text-card-foreground focus:outline-none focus:border-indigo-500"
          >
            <option value="">All Quality Tiers</option>
            <option value="EXCELLENT">EXCELLENT</option>
            <option value="ACCEPTABLE">ACCEPTABLE</option>
            <option value="POOR">POOR</option>
            <option value="UNUSABLE">UNUSABLE</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className={selectedRecord ? 'lg:col-span-7' : 'lg:col-span-12'}>
          <div className="glass-panel rounded-2xl overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-card/80 text-muted-foreground uppercase font-semibold border-b border-border">
                  <tr>
                    <th className="px-4 py-3">ID</th>
                    <th className="px-4 py-3">File</th>
                    <th className="px-4 py-3">Score</th>
                    <th className="px-4 py-3">Rating</th>
                    <th className="px-4 py-3">Issues</th>
                    <th className="px-4 py-3">Timestamp</th>
                    <th className="px-4 py-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/60 font-mono">
                  {filteredItems.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="px-4 py-8 text-center text-muted-foreground font-sans">
                        No inspection history records found.
                      </td>
                    </tr>
                  ) : (
                    filteredItems.map((item) => (
                      <tr
                        key={item.id}
                        className={`hover:bg-muted/40 cursor-pointer transition-colors ${
                          selectedRecord?.id === item.id ? 'bg-indigo-500/10' : ''
                        }`}
                        onClick={() => handleSelectRecord(item)}
                      >
                        <td className="px-4 py-3 text-muted-foreground">#{item.id}</td>
                        <td className="px-4 py-3 font-sans font-medium text-card-foreground">
                          {item.filename}
                        </td>
                        <td className="px-4 py-3 font-bold text-foreground">
                          {item.quality_score.toFixed(1)}
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                              item.quality_label === 'EXCELLENT'
                                ? 'bg-emerald-500/10 text-emerald-400'
                                : item.quality_label === 'ACCEPTABLE'
                                ? 'bg-blue-500/10 text-blue-400'
                                : item.quality_label === 'POOR'
                                ? 'bg-amber-500/10 text-amber-400'
                                : 'bg-rose-500/10 text-rose-400'
                            }`}
                          >
                            {item.quality_label}
                          </span>
                        </td>
                        <td className="px-4 py-3 font-sans text-popover-foreground">
                          {item.issue_count !== undefined
                            ? `${item.issue_count} detected`
                            : item.issues_summary && item.issues_summary.length > 0
                            ? `${item.issues_summary.length} detected`
                            : '0 detected'}
                        </td>
                        <td className="px-4 py-3 text-muted-foreground font-sans text-[11px]">
                          {item.created_at ? new Date(item.created_at).toLocaleTimeString() : '—'}
                        </td>
                        <td className="px-4 py-3 text-right" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => item.id && handleDelete(item.id)}
                            className="p-1 hover:bg-muted rounded text-muted-foreground hover:text-rose-400 transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {selectedRecord && (
          <div className="lg:col-span-5 space-y-6">
            <div className="glass-panel rounded-2xl p-5 relative">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-base font-bold text-foreground">{selectedRecord.filename}</h3>
                  <p className="text-xs text-muted-foreground">Record #{selectedRecord.id}</p>
                </div>
                <button
                  onClick={() => setSelectedRecord(null)}
                  className="p-1 text-muted-foreground hover:text-card-foreground hover:bg-muted rounded-lg"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <ScoreGauge
                score={selectedRecord.quality_score}
                label={selectedRecord.quality_label}
                processingTimeMs={selectedRecord.processing_time_ms}
              />

              <div className="mt-4 space-y-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground block">
                  Issues ({selectedRecord.issues ? selectedRecord.issues.length : 0})
                </span>
                {selectedRecord.issues && selectedRecord.issues.length > 0 ? (
                  selectedRecord.issues.map((issue, idx) => (
                    <IssueCard key={idx} issue={issue} />
                  ))
                ) : (
                  <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 text-xs font-medium">
                    ✓ Clean image: No issues detected.
                  </div>
                )}
              </div>

              {selectedRecord.stats && (
                <div className="mt-4">
                  <MetricsBreakdown stats={selectedRecord.stats} />
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
