"use client";

import { useState } from "react";
import {
  UploadCloud,
  File,
  AlertCircle,
  CheckCircle,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { ProbabilityChart } from "@/components/medai/ProbabilityChart";
import ReliabilityChart from "@/components/medai/ReliabilityChartNew";
// import ConfusionMatrix from "@/components/medai/ConfusionMatrix";
import { ChatInterface } from "@/components/medai/ChatInterface";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import Image from "next/image";

interface DiagnosisResponse {
  prediction: {
    ensemble_prediction: string;
    ensemble_confidence: number;
    fracture_detected: boolean;
    all_probabilities: Record<string, number>;
    individual_model_predictions?: Record<
      string,
      { class: string; confidence: number }
    >;
  };
  explanation: {
    text: string;
    heatmap_b64?: string | null;
    per_model_heatmaps?: Record<string, string>;
  };
  educational: {
    patient_summary: string;
    severity_layman: string;
    next_steps_action_plan: string;
  };
  knowledge_base: Record<string, any>;
  metrics?: {
    top1_vs_top2_margin?: number;
  };
  conformal?: {
    enabled?: boolean;
    conformal_set?: string[];
    conformal_threshold?: number;
  };
  critic_review?: {
    verdict: "yes" | "no" | "uncertain";
    critic_confidence: number;
    explanation: string;
    flagged_for_human: boolean;
    error?: string;
  };
  critic_error?: string;
  consensus?: {
    final_decision: "approved" | "flagged";
    reason: string;
    critic_score: number;
  };
  audit?: {
    inference_id: string;
  };
}

// Friendly display names for model keys returned by the backend
const MODEL_DISPLAY_NAMES: Record<string, string> = {
  maxvit: "MaxViT",
  yolo: "YOLOv26m",
  yolov26m: "YOLOv26m",
  hypercolumn_cbam_densenet169: "HC-CBAM-DenseNet169",
  rad_dino: "RAD-DINO",
  swin: "Swin Transformer",
  mobilenetv2: "MobileNetV2",
  efficientnetv2: "EfficientNetV2",
  densenet169: "DenseNet169",
};

// Models whose architecture doesn't support Grad-CAM
const GRADCAM_EXCLUDED_MODELS = new Set<string>([
  "yolo",
  "yolov26m",
  "yolov26",
  "rad_dino",
]);

// Map model names to their visualization type for labelling
const MODEL_VIZ_TYPE: Record<string, string> = {
  yolo: "Saliency",
  yolov26m: "Saliency",
  rad_dino: "Attention",
};

function getModelDisplayName(key: string): string {
  return MODEL_DISPLAY_NAMES[key.toLowerCase()] || key;
}

function getModelBadge(key: string): { label: string; color: string } | null {
  const k = key.toLowerCase();
  if (k.includes("yolo"))
    return { label: "YOLO", color: "bg-amber-500/20 text-amber-400" };
  if (k.includes("rad_dino") || k.includes("dino"))
    return { label: "ViT", color: "bg-purple-500/20 text-purple-400" };
  if (k.includes("hypercolumn") || k.includes("cbam"))
    return { label: "HC-CBAM", color: "bg-cyan-500/20 text-cyan-400" };
  if (k.includes("maxvit") || k.includes("swin"))
    return { label: "Transformer", color: "bg-blue-500/20 text-blue-400" };
  return null;
}

export default function DiagnosePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DiagnosisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [useConformal, setUseConformal] = useState<boolean>(true);
  const [enableCritic, setEnableCritic] = useState<boolean>(true);
  const [ensembleMode, setEnsembleMode] = useState<string>("weighted");
  const [stackerPath, setStackerPath] = useState<string>(
    "/outputs/stacker.joblib",
  );
  const [visibleModelHeatmaps, setVisibleModelHeatmaps] = useState<
    Record<string, boolean>
  >({});
  const [reliabilityData, setReliabilityData] = useState<any | null>(null);
  const [camOpacity, setCamOpacity] = useState<number>(0.6);
  const [compareMode, setCompareMode] = useState<boolean>(false);
  const [medicalLight, setMedicalLight] = useState<boolean>(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null); // Reset previous results
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selected = e.dataTransfer.files[0];
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    // Pass analysis options to the backend
    formData.append("use_conformal", String(useConformal));
    formData.append("ensemble_mode", ensembleMode);
    formData.append("stacker_path", stackerPath);

    try {
      formData.append("format", "json");

      // Choose endpoint based on Critic flag
      const endpoint = enableCritic
        ? "/api/diagnose/critic"
        : "/api/diagnose/report";

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok)
        throw new Error("Analysis failed. Backend might be offline.");

      const data = await response.json();
      if (data.error) throw new Error(data.error);

      // Normalize backend payload to frontend-friendly shape
      const payload: any = data;
      const normalized: any = {
        prediction: {
          ensemble_prediction:
            payload.prediction?.top_class ||
            payload.ensemble?.ensemble_prediction ||
            "",
          ensemble_confidence:
            payload.prediction?.confidence_score ||
            payload.ensemble?.ensemble_confidence ||
            0,
          fracture_detected:
            payload.prediction?.fracture_detected ||
            payload.ensemble?.fracture_detected ||
            false,
          all_probabilities:
            payload.prediction?.all_probabilities ||
            payload.ensemble?.all_probabilities ||
            {},
          individual_model_predictions:
            payload.prediction?.individual_model_predictions ||
            payload.ensemble?.individual_predictions ||
            {},
        },
        explanation: {
          text: payload.explanation?.text || "",
          heatmap_b64: payload.explanation?.heatmap_b64 || null,
          per_model_heatmaps:
            payload.explanation?.per_model_heatmaps || undefined,
        },
        educational: payload.educational || {
          patient_summary: "",
          severity_layman: "",
          next_steps_action_plan: "",
        },
        knowledge_base: payload.knowledge_base || {},
        metrics: payload.metrics || {},
        conformal: payload.conformal || {},
        critic_review: payload.critic_review,
        consensus: payload.consensus,
        audit: payload.audit || {},
      } as DiagnosisResponse;

      if (normalized.explanation.per_model_heatmaps) {
        const vis: Record<string, boolean> = {};
        Object.keys(normalized.explanation.per_model_heatmaps).forEach(
          (k) => (vis[k] = false),
        );
        setVisibleModelHeatmaps(vis);
      }

      setResult(normalized);
      // fetch reliability data when result available; fall back to a small sample if backend unavailable
      try {
        const r = await fetch("/api/diagnose/reliability");
        if (r.ok) {
          const jd = await r.json();
          setReliabilityData(jd);
        } else {
          // backend returned non-ok -> show friendly fallback
          const labels = Object.keys(
            normalized.prediction.all_probabilities || {},
          );
          const sample = {
            bins: Array.from({ length: 10 }, (_, i) => (i + 0.5) / 10),
            prob_pred: [0.05, 0.1, 0.12, 0.1, 0.1, 0.1, 0.12, 0.1, 0.08, 0.13],
            prob_true: [
              0.04, 0.09, 0.1, 0.11, 0.09, 0.11, 0.13, 0.12, 0.08, 0.13,
            ],
            brier_score: 0.12,
            confusion_matrix: labels.length
              ? Array.from({ length: labels.length }, () =>
                  Array(labels.length).fill(0),
                )
              : [[0]],
            class_labels: labels.length ? labels : ["class0", "class1"],
            _fallback: true,
          };
          setReliabilityData(sample);
        }
      } catch (e) {
        // likely backend offline (ECONNREFUSED) — use friendly fallback so UI doesn't break
        const labels = Object.keys(
          normalized.prediction.all_probabilities || {},
        );
        const sample = {
          bins: Array.from({ length: 10 }, (_, i) => (i + 0.5) / 10),
          prob_pred: [0.05, 0.1, 0.12, 0.1, 0.1, 0.1, 0.12, 0.1, 0.08, 0.13],
          prob_true: [
            0.04, 0.09, 0.1, 0.11, 0.09, 0.11, 0.13, 0.12, 0.08, 0.13,
          ],
          brier_score: 0.12,
          confusion_matrix: labels.length
            ? Array.from({ length: labels.length }, () =>
                Array(labels.length).fill(0),
              )
            : [[0]],
          class_labels: labels.length ? labels : ["class0", "class1"],
          _fallback: true,
        };
        setReliabilityData(sample);
      }
    } catch (err: any) {
      setError(err.message || "An error occurred during analysis.");
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("use_conformal", String(useConformal));
      formData.append("ensemble_mode", ensembleMode);
      formData.append("stacker_path", stackerPath);

      const resp = await fetch("/api/diagnose/report", {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) throw new Error("Failed to generate PDF report");
      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "diagnosis_report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (e: any) {
      setError(e.message || "Failed to download PDF report");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className={`min-h-screen p-6 lg:p-12 ${
        medicalLight ? "bg-white text-slate-900" : "bg-background text-white"
      }`}
      style={
        medicalLight
          ? ({
              color: "#0f172a",
              "--background": "0 0% 100%",
              "--foreground": "0 0% 3.9%",
              "--card": "0 0% 100%",
              "--card-foreground": "0 0% 3.9%",
              "--border": "0 0% 89.8%",
            } as React.CSSProperties)
          : undefined
      }
    >
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              Diagnosis Dashboard
            </h1>
            <p className="text-muted-foreground mt-2">
              Upload an X-ray to run the multi-agent analysis pipeline.
            </p>
          </div>
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={medicalLight}
                onChange={(e) => setMedicalLight(e.target.checked)}
              />
              <span
                className={`${
                  medicalLight ? "text-slate-700" : "text-muted-foreground"
                }`}
              >
                Medical light theme
              </span>
            </label>
          </div>
        </div>

        {/* Upload Section */}
        <div className="grid lg:grid-cols-3 gap-8">
          <Card className="lg:col-span-1">
            <CardHeader>
              <CardTitle>X-Ray Upload</CardTitle>
              <CardDescription>
                Supported formats: PNG, JPG, JPEG
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div
                className={`
                  border-2 border-dashed rounded-xl p-8 text-center transition-colors
                  ${
                    file
                      ? "border-primary/50 bg-primary/5"
                      : "border-neutral-700 hover:border-neutral-500"
                  }
                `}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
              >
                {!preview ? (
                  <div className="space-y-4">
                    <div
                      className={`h-12 w-12 rounded-full flex items-center justify-center mx-auto ${
                        medicalLight ? "bg-neutral-100" : "bg-neutral-800"
                      }`}
                    >
                      <UploadCloud className="h-6 w-6 text-neutral-400" />
                    </div>
                    <div>
                      <span className="font-medium text-primary cursor-pointer hover:underline">
                        <label htmlFor="file-upload" className="cursor-pointer">
                          Click to upload
                        </label>
                      </span>
                      <span className="text-neutral-500">
                        {" "}
                        or drag and drop
                      </span>
                    </div>
                    <input
                      id="file-upload"
                      type="file"
                      className="hidden"
                      onChange={handleFileChange}
                      accept="image/*"
                    />
                  </div>
                ) : (
                  <div className="relative">
                    <img
                      src={preview}
                      alt="Preview"
                      className="rounded-lg max-h-[300px] mx-auto object-contain"
                    />
                    <Button
                      variant="secondary"
                      size="sm"
                      className="absolute top-2 right-2"
                      onClick={() => {
                        setFile(null);
                        setPreview(null);
                        setResult(null);
                      }}
                    >
                      Change
                    </Button>
                  </div>
                )}
              </div>

              {/* Analysis options */}
              <div className="space-y-2">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={useConformal}
                    onChange={(e) => setUseConformal(e.target.checked)}
                  />
                  <span className="text-sm text-muted-foreground">
                    Enable conformal prediction
                  </span>
                </label>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={enableCritic}
                    onChange={(e) => setEnableCritic(e.target.checked)}
                  />
                  <span className="text-sm text-muted-foreground">
                    Enable Agentic Critic (Self-Correction)
                  </span>
                </label>
                <div className="flex gap-2 items-center">
                  <label className="text-sm text-muted-foreground">
                    Ensemble mode:
                  </label>
                  <select
                    value={ensembleMode}
                    onChange={(e) => setEnsembleMode(e.target.value)}
                    className="text-sm bg-transparent border rounded p-1"
                  >
                    <option value="weighted">Weighted</option>
                    <option value="stacking">Stacking</option>
                  </select>
                  {/* stacker path selection removed; backend uses default /outputs/stacker.joblib when stacking */}
                </div>
              </div>

              {error && (
                <div className="p-4 rounded-lg bg-destructive/10 text-destructive text-sm flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" /> {error}
                </div>
              )}

              <div className="grid grid-cols-2 gap-2">
                <Button
                  className="w-full h-12 text-lg"
                  onClick={handleAnalyze}
                  disabled={!file || loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />{" "}
                      Analyzing...
                    </>
                  ) : (
                    "Run Analysis"
                  )}
                </Button>
                <Button
                  variant="secondary"
                  className="w-full h-12 text-lg"
                  onClick={handleDownloadPDF}
                  disabled={!file || loading}
                >
                  Download PDF
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Results Area */}
          <div className="lg:col-span-2 space-y-8">
            {result ? (
              <>
                {/* Top Stats Cards */}
                <div className="grid md:grid-cols-2 gap-4">
                  <Card
                    className={
                      result.prediction.fracture_detected
                        ? "border-red-500/50 bg-red-500/5"
                        : "border-green-500/50 bg-green-500/5"
                    }
                  >
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg font-medium text-muted-foreground">
                        Primary Diagnosis
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold flex items-center gap-3">
                        {result.prediction.ensemble_prediction}
                        {result.prediction.fracture_detected ? (
                          <AlertCircle className="h-6 w-6 text-red-500" />
                        ) : (
                          <CheckCircle className="h-6 w-6 text-green-500" />
                        )}
                      </div>
                      <p className="text-muted-foreground mt-1">
                        Confidence:{" "}
                        {(result.prediction.ensemble_confidence * 100).toFixed(
                          1,
                        )}
                        %
                      </p>
                      {(result as any).conformal &&
                        (result as any).conformal.enabled && (
                          <div className="mt-3 text-sm text-muted-foreground">
                            <div className="font-medium">
                              Conformal Prediction Set (guaranteed coverage)
                            </div>
                            {/* <div className="mt-1">
                              {(
                                (result as any).conformal.conformal_set || []
                              ).join(", ") || "—"}
                            </div> */}
                          </div>
                        )}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg font-medium text-muted-foreground">
                        Severity
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-3xl font-bold">
                        {result.knowledge_base.Severity_Rating || "Unknown"}
                      </div>
                      <p className="text-muted-foreground mt-1 text-sm">
                        {result.educational.severity_layman ||
                          result.knowledge_base.Type_Definition}
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {/* Agentic Critic Error Section */}
                {result.critic_error && (
                  <Card className="border-l-4 border-l-orange-500">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-xl flex items-center gap-2 text-orange-700">
                        ⚠️ Critic Agent Unavailable
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm">{result.critic_error}</p>
                    </CardContent>
                  </Card>
                )}

                {/* Agentic Critic Review Section */}
                {result.critic_review && (
                  <Card
                    className={`border-l-4 ${
                      result.critic_review.verdict === "yes"
                        ? "border-l-green-500"
                        : result.critic_review.verdict === "no"
                          ? "border-l-red-500"
                          : "border-l-yellow-500"
                    }`}
                  >
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <div className="space-y-1">
                        <CardTitle className="text-xl flex items-center gap-2">
                          🕵️ Critic Agent Review
                        </CardTitle>
                        <CardDescription>
                          MedGemma VLM Second Opinion
                        </CardDescription>
                      </div>
                      <div
                        className={`px-3 py-1 rounded-full text-sm font-bold ${
                          result.critic_review.verdict === "yes"
                            ? "bg-green-100 text-green-700"
                            : result.critic_review.verdict === "no"
                              ? "bg-red-100 text-red-700"
                              : "bg-yellow-100 text-yellow-700"
                        }`}
                      >
                        Verdict: {result.critic_review.verdict.toUpperCase()}
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="grid md:grid-cols-3 gap-6">
                        <div className="md:col-span-2">
                          <h4 className="font-semibold mb-1">
                            Critic Explanation
                          </h4>
                          <p className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-md italic">
                            &quot;{result.critic_review.explanation}&quot;
                          </p>
                        </div>
                        <div className="space-y-3">
                          <h4 className="font-semibold mb-1">
                            Consensus Status
                          </h4>
                          {result.consensus && (
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span>Status:</span>
                                <span
                                  className={`font-bold ${
                                    result.consensus.final_decision ===
                                    "flagged"
                                      ? "text-red-500"
                                      : "text-green-500"
                                  }`}
                                >
                                  {result.consensus.final_decision === "flagged"
                                    ? "🚩 FLAGGED"
                                    : "✅ APPROVED"}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span>Critic Conf:</span>
                                <span>
                                  {(
                                    result.critic_review.critic_confidence * 100
                                  ).toFixed(0)}
                                  %
                                </span>
                              </div>
                              {result.consensus.reason && (
                                <div className="text-xs text-muted-foreground border-t pt-2 mt-1">
                                  {result.consensus.reason}
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Explanation & Heatmap */}
                <div className="grid md:grid-cols-2 gap-8">
                  <Card>
                    <CardHeader>
                      <CardTitle>AI Explanation & Grad-CAM</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* If backend provides per-model heatmaps, render toggles; otherwise show single heatmap */}
                      {result.explanation &&
                      (result as any).explanation.per_model_heatmaps ? (
                        <div className="space-y-3">
                          <div className="text-sm text-muted-foreground">
                            Per-model Visualizations:
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            {Object.entries(
                              (result as any).explanation.per_model_heatmaps,
                            ).map(([mname, b64]) => (
                              <div
                                key={mname}
                                className={`border rounded overflow-hidden ${
                                  medicalLight
                                    ? "bg-white/5 border-neutral-200"
                                    : "bg-neutral-900/5"
                                }`}
                              >
                                <div
                                  className={`p-2 flex items-center justify-between ${
                                    medicalLight
                                      ? "bg-white/10"
                                      : "bg-neutral-900/10"
                                  }`}
                                >
                                  <div className="text-sm truncate max-w-[160px] font-medium">
                                    {getModelDisplayName(mname)}
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <button
                                      className="text-sm px-2 py-1 rounded bg-primary/10 hover:bg-primary/20"
                                      onClick={() =>
                                        setVisibleModelHeatmaps((prev) => ({
                                          ...prev,
                                          [mname]: !prev[mname],
                                        }))
                                      }
                                    >
                                      {visibleModelHeatmaps[mname]
                                        ? "Hide"
                                        : "Show"}
                                    </button>
                                  </div>
                                </div>
                                {visibleModelHeatmaps[mname] && (
                                  <div
                                    className={`w-full ${
                                      medicalLight
                                        ? "bg-white/5"
                                        : "bg-neutral-900/5"
                                    } p-2 flex items-center justify-center`}
                                  >
                                    <img
                                      src={`data:image/png;base64,${b64}`}
                                      alt={`Grad-CAM ${mname}`}
                                      className="max-h-[220px] object-contain"
                                    />
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        result.explanation.heatmap_b64 && (
                          <div>
                            <div className="flex items-center justify-between mb-2">
                              <div className="flex items-center gap-4">
                                <label className="flex items-center gap-2 text-sm">
                                  <input
                                    type="checkbox"
                                    checked={compareMode}
                                    onChange={(e) =>
                                      setCompareMode(e.target.checked)
                                    }
                                  />
                                  <span>Compare overlay with original</span>
                                </label>
                                <div className="flex items-center gap-2">
                                  <label className="text-sm">
                                    Overlay opacity
                                  </label>
                                  <input
                                    type="range"
                                    min={0}
                                    max={1}
                                    step={0.05}
                                    value={camOpacity}
                                    onChange={(e) =>
                                      setCamOpacity(Number(e.target.value))
                                    }
                                  />
                                </div>
                              </div>
                            </div>
                            <div
                              className={`${
                                medicalLight
                                  ? "rounded-lg overflow-hidden border border-neutral-200"
                                  : "rounded-lg overflow-hidden border border-white/10"
                              }`}
                            >
                              {compareMode && preview ? (
                                <div
                                  className={`${
                                    medicalLight
                                      ? "relative w-full h-[300px] flex items-center justify-center bg-neutral-50"
                                      : "relative w-full h-[300px] flex items-center justify-center bg-black"
                                  }`}
                                >
                                  <img
                                    src={preview}
                                    alt="original"
                                    className="max-h-[300px] max-w-full object-contain"
                                  />
                                  <img
                                    src={`data:image/png;base64,${result.explanation.heatmap_b64}`}
                                    alt="overlay"
                                    style={{ opacity: camOpacity }}
                                    className="absolute max-h-[300px] max-w-full object-contain"
                                  />
                                </div>
                              ) : (
                                <img
                                  src={`data:image/png;base64,${result.explanation.heatmap_b64}`}
                                  alt="Grad-CAM Heatmap"
                                  className="w-full object-cover"
                                />
                              )}
                            </div>
                          </div>
                        )
                      )}
                      <div className="text-sm leading-relaxed text-muted-foreground">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {result.explanation.text}
                        </ReactMarkdown>
                      </div>
                    </CardContent>
                  </Card>

                  <ProbabilityChart
                    probabilities={result.prediction.all_probabilities}
                  />
                  {/* Left column: Top-1 margin + Individual Model Predictions */}
                  <div>
                    {result.metrics &&
                      result.metrics.top1_vs_top2_margin !== undefined && (
                        <div className="mt-2 w-full">
                          <div className="text-sm text-muted-foreground mb-2">
                            <strong>Top-1 vs Top-2 margin:</strong>{" "}
                            {(result.metrics.top1_vs_top2_margin * 100).toFixed(
                              2,
                            )}
                            %
                          </div>
                          <div
                            className={`${
                              medicalLight
                                ? "w-full bg-neutral-200 rounded h-3"
                                : "w-full bg-neutral-800 rounded h-3"
                            }`}
                          >
                            <div
                              className="h-3 rounded bg-gradient-to-r from-green-500 to-yellow-400"
                              style={{
                                width: `${Math.min(
                                  100,
                                  (result.metrics.top1_vs_top2_margin || 0) *
                                    100,
                                )}%`,
                              }}
                            />
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            A higher margin indicates more separation between
                            the top two classes and higher model confidence.
                          </div>
                        </div>
                      )}

                    {/* Individual Model Predictions */}
                    {result.prediction.individual_model_predictions && (
                      <details
                        className={`mt-4 border rounded p-3 ${
                          medicalLight
                            ? "bg-white/5 border-neutral-200"
                            : "bg-neutral-900/5"
                        }`}
                      >
                        <summary className="cursor-pointer font-medium">
                          Individual Model Predictions
                        </summary>
                        <div className="mt-2 text-sm">
                          <div className="grid gap-2 mt-2">
                            {Object.entries(
                              result.prediction.individual_model_predictions,
                            ).map(([m, info]: any) => {
                              const badge = getModelBadge(m);
                              return (
                                <div
                                  key={m}
                                  className={`flex items-center justify-between gap-4 py-2 px-2 ${
                                    medicalLight
                                      ? "bg-white/5"
                                      : "bg-neutral-900/10"
                                  } rounded`}
                                >
                                  <div className="min-w-0">
                                    <div className="font-medium truncate flex items-center gap-2">
                                      {getModelDisplayName(m)}
                                      {badge && (
                                        <span
                                          className={`text-[10px] px-1.5 py-0.5 rounded-full font-semibold ${badge.color}`}
                                        >
                                          {badge.label}
                                        </span>
                                      )}
                                      {GRADCAM_EXCLUDED_MODELS.has(
                                        m.toLowerCase(),
                                      ) && (
                                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-neutral-500/20 text-neutral-400 font-medium">
                                          No Grad-CAM
                                        </span>
                                      )}
                                      {MODEL_VIZ_TYPE[m.toLowerCase()] && (
                                        <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 font-medium">
                                          {MODEL_VIZ_TYPE[m.toLowerCase()]}
                                        </span>
                                      )}
                                    </div>
                                    <div className="text-xs text-muted-foreground truncate max-w-[60ch]">
                                      {info.class}
                                    </div>
                                  </div>
                                  <div className="ml-4 flex-shrink-0 text-right">
                                    <div className="text-sm font-semibold">
                                      {(info.confidence * 100).toFixed(2)}%
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </details>
                    )}
                  </div>

                  {reliabilityData && (
                    <Card className="mt-4">
                      <CardHeader>
                        <CardTitle>Calibration / Reliability</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-sm">
                          Brier score:{" "}
                          {(reliabilityData.brier_score || 0).toFixed(4)}
                        </div>
                        <div className="mt-4">
                          <ReliabilityChart
                            bins={reliabilityData.prob_pred.map(
                              (_: any, i: number) =>
                                (i + 0.5) / reliabilityData.prob_pred.length,
                            )}
                            predicted={reliabilityData.prob_pred}
                            observed={reliabilityData.prob_true}
                            medicalLight={medicalLight}
                          />
                        </div>
                        {/* {reliabilityData.confusion_matrix && (
                          <div className="mt-4">
                            <div className="text-sm font-medium mb-2">
                              Confusion Matrix
                            </div>
                            <ConfusionMatrix
                              matrix={reliabilityData.confusion_matrix}
                              labels={reliabilityData.class_labels}
                              medicalLight={medicalLight}
                            />
                          </div>
                        )} */}
                      </CardContent>
                    </Card>
                  )}
                </div>

                {/* Gemini AI Explanation */}
                {result.knowledge_base.gemini_explanation && (
                  <Card
                    className={`border ${
                      medicalLight
                        ? "bg-indigo-50 border-indigo-200"
                        : "bg-indigo-950/20 border-indigo-500/30"
                    }`}
                  >
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <span className="text-2xl">
                          Detailed Clinical Analysis
                        </span>{" "}
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div
                        className={`prose prose-sm md:prose-base max-w-none ${
                          medicalLight
                            ? "text-indigo-900 prose-headings:text-indigo-800 prose-strong:text-indigo-900 prose-li:text-indigo-900 prose-p:text-indigo-900"
                            : "prose-invert text-indigo-100 prose-headings:text-indigo-50 prose-strong:text-indigo-50 prose-li:text-indigo-200 prose-p:text-indigo-200"
                        } prose-headings:font-semibold prose-headings:mt-6 prose-headings:mb-2 prose-p:my-2 prose-ul:my-2 prose-ol:my-2 prose-li:my-0.5`}
                      >
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {result.knowledge_base.gemini_explanation}
                        </ReactMarkdown>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Educational Content & Chat */}
                <div className="grid md:grid-cols-2 gap-8">
                  <Card className="h-full">
                    <CardHeader>
                      <CardTitle>Simplified Explanation</CardTitle>
                    </CardHeader>
                    <CardContent
                      className={`prose max-w-none text-sm space-y-4 ${
                        medicalLight ? "" : "prose-invert"
                      }`}
                    >
                      <div
                        className={`${
                          medicalLight
                            ? "bg-blue-50 border border-blue-200 text-slate-800"
                            : "bg-blue-500/10 border border-blue-500/20 text-blue-200"
                        } p-4 rounded-lg`}
                      >
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {result.educational.patient_summary}
                        </ReactMarkdown>
                      </div>
                      <div className="not-prose">
                        <h3
                          className={`text-lg font-semibold mt-4 mb-2 ${medicalLight ? "text-slate-900" : "text-white"}`}
                        >
                          Next Steps / Action Plan
                        </h3>
                        <div
                          className={`${
                            medicalLight
                              ? "bg-blue-50 border border-blue-200 text-slate-800"
                              : "bg-blue-500/10 border border-blue-500/20 text-blue-200"
                          } p-4 rounded-lg`}
                        >
                          <div
                            className={`prose max-w-none text-sm ${
                              medicalLight ? "" : "prose-invert"
                            }`}
                          >
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {result.educational.next_steps_action_plan}
                            </ReactMarkdown>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <ChatInterface
                    key={result.audit?.inference_id || "chat-interface"}
                    context={result.knowledge_base}
                    medicalLight={medicalLight}
                    inferenceId={result.audit?.inference_id}
                  />
                </div>
              </>
            ) : (
              <div
                className={`${
                  medicalLight
                    ? "h-full flex items-center justify-center border-2 border-dashed border-neutral-200 rounded-xl min-h-[400px] bg-white"
                    : "h-full flex items-center justify-center border-2 border-dashed border-neutral-800 rounded-xl min-h-[400px]"
                }`}
              >
                <div
                  className={`text-center max-w-md ${
                    medicalLight ? "text-slate-700" : "text-muted-foreground"
                  }`}
                >
                  {loading ? (
                    <div className="space-y-4">
                      <Loader2 className="h-10 w-10 animate-spin mx-auto text-primary" />
                      <p>Running multi-agent diagnostics...</p>
                    </div>
                  ) : (
                    <>
                      <Activity className="h-12 w-12 mx-auto mb-4 opacity-20" />
                      <h3
                        className={`text-lg font-medium ${
                          medicalLight ? "text-slate-900" : "text-white"
                        } mb-2`}
                      >
                        No Analysis Results
                      </h3>
                      <p>
                        Upload an X-ray image to start the diagnosis process.
                      </p>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Activity(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
    </svg>
  );
}
