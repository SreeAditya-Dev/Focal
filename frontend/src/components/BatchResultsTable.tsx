import React from 'react';
import { Download, CheckCircle2, AlertTriangle, XCircle, FileText } from 'lucide-react';
import { BatchAnalysisResponse, AnalysisResponse } from '../api/types';

interface BatchResultsTableProps {
  batchData: BatchAnalysisResponse;
  onSelectResult: (res: AnalysisResponse) => void;
}

export const BatchResultsTable: React.FC<BatchResultsTableProps> = ({
  batchData,
  onSelectResult,
}) => {
  const exportCSV = () => {
    const headers = ['Filename', 'Quality Score', 'Quality Label', 'Issues Detected', 'Inference Time (ms)'];
    const rows = batchData.results.map((r) => [
      r.filename,
      r.quality_score.toFixed(1),
      r.quality_label,
      r.issues.map((i) => `${i.type} (${i.severity})`).join('; ') || 'None',
      r.processing_time_ms.toFixed(1),
    ]);

    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.map((val) => `"${val}"`).join(','))].join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `focal_batch_report_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getLabelBadge = (label: string) => {
    switch (label) {
      case 'EXCELLENT':
        return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
      case 'ACCEPTABLE':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'POOR':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
      case 'UNUSABLE':
      default:
        return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-base font-bold text-foreground">Batch Inspection Summary</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Processed {batchData.successful} of {batchData.total} files in {(batchData.total_time_ms / 1000).toFixed(2)}s
          </p>
        </div>

        <button
          onClick={exportCSV}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md shadow-indigo-600/30 transition-all"
        >
          <Download className="w-3.5 h-3.5" />
          Export CSV Report
        </button>
      </div>

      <div className="overflow-x-auto rounded-xl border border-border">
        <table className="w-full text-left text-xs">
          <thead className="bg-card/80 text-muted-foreground uppercase font-semibold border-b border-border">
            <tr>
              <th className="px-4 py-3">File</th>
              <th className="px-4 py-3">Score</th>
              <th className="px-4 py-3">Rating</th>
              <th className="px-4 py-3">Detected Degradations</th>
              <th className="px-4 py-3">Latency</th>
              <th className="px-4 py-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/60 font-mono">
            {batchData.results.map((r, i) => (
              <tr key={i} className="hover:bg-muted/30 transition-colors">
                <td className="px-4 py-3 font-sans font-medium text-card-foreground">{r.filename}</td>
                <td className="px-4 py-3 font-bold text-foreground">{r.quality_score.toFixed(1)}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold border uppercase ${getLabelBadge(r.quality_label)}`}>
                    {r.quality_label}
                  </span>
                </td>
                <td className="px-4 py-3 font-sans text-popover-foreground">
                  {r.issues.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {r.issues.map((iss, j) => (
                        <span key={j} className="bg-muted px-1.5 py-0.5 rounded text-[11px] text-indigo-300">
                          {iss.type} ({iss.severity})
                        </span>
                      ))}
                    </div>
                  ) : (
                    <span className="text-emerald-400">Clean</span>
                  )}
                </td>
                <td className="px-4 py-3 text-muted-foreground">{r.processing_time_ms.toFixed(0)}ms</td>
                <td className="px-4 py-3 text-right">
                  <button
                    onClick={() => onSelectResult(r)}
                    className="text-xs text-indigo-400 hover:text-indigo-300 font-sans font-semibold"
                  >
                    View Details →
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

