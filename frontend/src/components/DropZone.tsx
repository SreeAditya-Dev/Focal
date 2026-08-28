import React, { useState, useRef } from "react";
import { UploadCloud, Image as ImageIcon, Layers, FileX, CheckCircle, Sparkles } from "lucide-react";

interface DropZoneProps {
  onSelectSingle: (file: File) => void;
  onSelectBatch: (files: File[]) => void;
  isProcessing: boolean;
  mode: "single" | "batch";
  onModeChange: (mode: "single" | "batch") => void;
}

export const DropZone: React.FC<DropZoneProps> = ({
  onSelectSingle,
  onSelectBatch,
  isProcessing,
  mode,
  onModeChange,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [batchFiles, setBatchFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(Array.from(e.target.files));
    }
  };

  const processFiles = (files: File[]) => {
    const validFiles = files.filter((f) =>
      /\.(jpe?g|png|webp|bmp|tiff)$/i.test(f.name)
    );

    if (validFiles.length === 0) {
      alert("Please select valid image files (JPEG, PNG, WebP, BMP, TIFF).");
      return;
    }

    if (mode === "single" || validFiles.length === 1) {
      const file = validFiles[0];
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      onSelectSingle(file);
    } else {
      setBatchFiles(validFiles);
      onSelectBatch(validFiles);
    }
  };

  // Helper to generate a test image programmatically for 1-click testing
  const createSampleImage = (type: string) => {
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Draw background
    ctx.fillStyle = type === "dark" ? "#111827" : type === "bright" ? "#fef08a" : "#3b82f6";
    ctx.fillRect(0, 0, 640, 480);

    if (type === "clean") {
      // Crisp sharp geometric shapes
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 32px 'Plus Jakarta Sans', sans-serif";
      ctx.fillText("FOCAL TEST IMAGE — CLEAN", 80, 240);
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 6;
      ctx.strokeRect(50, 50, 540, 380);
    } else if (type === "blur") {
      ctx.filter = "blur(12px)";
      ctx.fillStyle = "#f43f5e";
      ctx.beginPath();
      ctx.arc(320, 240, 100, 0, Math.PI * 2);
      ctx.fill();
      ctx.filter = "none";
    } else if (type === "noise") {
      ctx.fillStyle = "#64748b";
      ctx.fillRect(0, 0, 640, 480);
      const imgData = ctx.getImageData(0, 0, 640, 480);
      for (let i = 0; i < imgData.data.length; i += 4) {
        const noise = (Math.random() - 0.5) * 160;
        imgData.data[i] += noise;
        imgData.data[i + 1] += noise;
        imgData.data[i + 2] += noise;
      }
      ctx.putImageData(imgData, 0, 0);
    }

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `sample_${type}.jpg`, { type: "image/jpeg" });
        processFiles([file]);
      }
    }, "image/jpeg", 0.95);
  };

  return (
    <div className="w-full">
      {/* Mode Switcher */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => onModeChange("single")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              mode === "single"
                ? "bg-slate-800 text-indigo-400 border border-indigo-500/20 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <ImageIcon className="h-3.5 w-3.5" />
            <span>Single Image</span>
          </button>
          <button
            onClick={() => onModeChange("batch")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
              mode === "batch"
                ? "bg-slate-800 text-indigo-400 border border-indigo-500/20 shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Layers className="h-3.5 w-3.5" />
            <span>Batch Upload</span>
          </button>
        </div>

        {/* Preset Sample Images */}
        <div className="flex items-center gap-1.5 overflow-x-auto text-xs">
          <span className="text-[11px] font-medium text-slate-500 hidden sm:inline">Try preset:</span>
          <button
            onClick={() => createSampleImage("clean")}
            disabled={isProcessing}
            className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-[11px] text-slate-300 hover:border-emerald-500/40 hover:text-emerald-400 transition-colors"
          >
            ✨ Clean
          </button>
          <button
            onClick={() => createSampleImage("blur")}
            disabled={isProcessing}
            className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-[11px] text-slate-300 hover:border-rose-500/40 hover:text-rose-400 transition-colors"
          >
            💨 Blurry
          </button>
          <button
            onClick={() => createSampleImage("noise")}
            disabled={isProcessing}
            className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-[11px] text-slate-300 hover:border-amber-500/40 hover:text-amber-400 transition-colors"
          >
            📻 Noisy
          </button>
        </div>
      </div>

      {/* Main Upload Drop Box */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`relative group cursor-pointer overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-300 p-8 text-center flex flex-col items-center justify-center min-h-[220px] ${
          isDragOver
            ? "border-indigo-500 bg-indigo-500/10 scale-[1.008]"
            : "border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/70"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff"
          multiple={mode === "batch"}
          onChange={handleFileInputChange}
          className="hidden"
        />

        {/* Ambient Glow */}
        <div className="absolute -top-24 -left-24 w-48 h-48 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none group-hover:bg-indigo-500/20 transition-all" />

        <div className="h-14 w-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
          <UploadCloud className="h-7 w-7 text-indigo-400 group-hover:text-indigo-300" />
        </div>

        <h3 className="text-base font-bold text-slate-200 mb-1">
          {mode === "single" ? "Drop your image here" : "Drop multiple images for batch evaluation"}
        </h3>
        <p className="text-xs text-slate-400 max-w-sm mb-4">
          Drag and drop your file or <span className="text-indigo-400 font-medium underline underline-offset-2">browse computer</span>
        </p>

        <div className="flex items-center gap-3 text-[11px] font-mono text-slate-500">
          <span className="px-2 py-0.5 rounded bg-slate-800/80">JPEG</span>
          <span className="px-2 py-0.5 rounded bg-slate-800/80">PNG</span>
          <span className="px-2 py-0.5 rounded bg-slate-800/80">WEBP</span>
          <span className="px-2 py-0.5 rounded bg-slate-800/80">MAX 25MB</span>
        </div>
      </div>
    </div>
  );
};

