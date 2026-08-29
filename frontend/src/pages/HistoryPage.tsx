import React, { useState, useEffect } from 'react';
import { getHistory, deleteHistoryItem, clearHistory } from '../api/client';
import { AnalysisResponse } from '../api/types';
import { Trash2, Search, Filter, RefreshCw, Eye, Calendar, FileText } from 'lucide-react';
import { ScoreGauge } from '../components/ScoreGauge';
import { IssueCard } from '../components/IssueCard';
import { MetricsBreakdown } from '../components/MetricsBreakdown';

export const HistoryPage: React.FC = () => {
  const [historyItems, setHistoryItems] = useState<AnalysisResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [labelFilter, setLabelFilter] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<AnalysisResponse | null>(null);

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const res = await getHistory(page, 20, labelFilter || undefined);
      setHistoryItems(res.items);
      setTotal(res.total);
    } catch (err) {
      console.error('Failed to load history', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [page, labelFilter]);

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

  const filteredItems = historyItems.filter((item) =>
    item.filename.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100">Audit History Log</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Persisted inspection logs stored in database ({total} total analyses)
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={fetchHistory}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-xs font-semibold text-slate-300 transition-colors"
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
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by filename..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full bg-slate-900/80 border border-slate-800 rounded-xl pl-9 pr-4 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors"
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <select
            value={labelFilter}
            onChange={(e) => {
              setLabelFilter(e.target.value);
              setPage(1);
            }}
            className="bg-slate-900/80 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
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
                <thead className="bg-slate-900/80 text-slate-400 uppercase font-semibold border-b border-slate-800">
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
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {filteredItems.length === 0 ? (
                    <tr>
                      <td colSpan={7} className="px-4 py-8 text-center text-slate-500 font-sans">
                        No inspection history records found.
                      </td>
                    </tr>
                  ) : (
                    filteredItems.map((item) => (
                      <tr
                        key={item.id}
                        className={`hover:bg-slate-800/40 cursor-pointer transition-colors ${
                          selectedRecord?.id === item.id ? 'bg-indigo-500/10' : ''
                        }`}
                        onClick={() => setSelectedRecord(item)}
                      >
                        <td className="px-4 py-3 text-slate-500">#{item.id}</td>
                        <td className="px-4 py-3 font-sans font-medium text-slate-200">
                          {item.filename}
                        </td>
                        <td className="px-4 py-3 font-bold text-slate-100">
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
                        <td className="px-4 py-3 font-sans text-slate-300">
                          {item.issues.length} detected
                        </td>
                        <td className="px-4 py-3 text-slate-500 font-sans text-[11px]">
                          {item.created_at ? new Date(item.created_at).toLocaleTimeString() : '—'}
                        </td>
                        <td className="px-4 py-3 text-right" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => item.id && handleDelete(item.id)}
                            className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-rose-400 transition-colors"
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
                  <h3 className="text-base font-bold text-slate-100">{selectedRecord.filename}</h3>
                  <p className="text-xs text-slate-400">Record #{selectedRecord.id}</p>
                </div>
                <button
                  onClick={() => setSelectedRecord(null)}
                  className="text-xs text-slate-400 hover:text-slate-200"
                >
                  ✕ Close
                </button>
              </div>

              <ScoreGauge
                score={selectedRecord.quality_score}
                label={selectedRecord.quality_label}
                processingTimeMs={selectedRecord.processing_time_ms}
              />

              <div className="mt-4 space-y-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block">
                  Issues ({selectedRecord.issues.length})
                </span>
                {selectedRecord.issues.map((issue, idx) => (
                  <IssueCard key={idx} issue={issue} />
                ))}
              </div>

              <div className="mt-4">
                <MetricsBreakdown stats={selectedRecord.stats} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

