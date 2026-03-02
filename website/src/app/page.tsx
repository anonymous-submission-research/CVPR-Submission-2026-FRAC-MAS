import Link from "next/link";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  ExternalLink,
  Github,
  FileText,
  ChevronRight,
} from "lucide-react";

// ── shared typography helpers ─────────────────────────────────────
const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <p className="text-xs font-semibold tracking-widest uppercase text-teal-400 mb-2">
    {children}
  </p>
);

const SectionTitle = ({ children }: { children: React.ReactNode }) => (
  <h2 className="text-2xl md:text-3xl font-bold text-slate-100">{children}</h2>
);

const FigCaption = ({
  num,
  children,
}: {
  num: string;
  children: React.ReactNode;
}) => (
  <p className="mt-3 text-sm text-slate-400 leading-relaxed">
    <span className="font-semibold text-slate-300">Figure {num}.</span>{" "}
    {children}
  </p>
);

const TableCaption = ({
  num,
  children,
}: {
  num: string;
  children: React.ReactNode;
}) => (
  <p className="mb-3 text-sm text-slate-400 leading-relaxed text-center">
    <span className="font-semibold text-slate-300">Table {num}.</span>{" "}
    {children}
  </p>
);

// ── stat card ─────────────────────────────────────────────────────
const StatCard = ({
  value,
  label,
  sub,
}: {
  value: string;
  label: string;
  sub?: string;
}) => (
  <div className="border border-slate-700 bg-slate-900/60 rounded-lg p-5 text-center space-y-1">
    <p className="text-3xl font-extrabold text-teal-400">{value}</p>
    <p className="text-sm font-medium text-slate-200">{label}</p>
    {sub && <p className="text-xs text-slate-500">{sub}</p>}
  </div>
);

// ── agent card ───────────────────────────────────────────────────
const AgentCard = ({
  number,
  name,
  role,
  desc,
}: {
  number: string;
  name: string;
  role: string;
  desc: string;
}) => (
  <div className="border border-slate-700 bg-slate-900/60 rounded-lg p-5 space-y-3">
    <div className="flex items-start gap-3">
      <span className="flex-shrink-0 h-7 w-7 rounded bg-teal-500/15 border border-teal-500/30 flex items-center justify-center text-xs font-bold text-teal-400">
        A{number}
      </span>
      <div>
        <p className="font-semibold text-slate-100">{name}</p>
        <p className="text-xs text-teal-400 font-medium">{role}</p>
      </div>
    </div>
    <p className="text-sm text-slate-400 leading-relaxed">{desc}</p>
  </div>
);

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-slate-950 text-slate-300">
      {/* ── Header ── */}
      <header className="px-6 lg:px-12 h-16 flex items-center justify-between border-b border-slate-800 bg-slate-950/90 backdrop-blur-sm sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold tracking-widest uppercase text-teal-400 border border-teal-500/30 bg-teal-500/10 px-2 py-0.5 rounded">
            CVPR 2026
          </span>
          <span className="hidden sm:block text-sm font-semibold text-slate-200">
            MACDSS
          </span>
        </div>
        <nav className="hidden md:flex gap-6 text-sm text-slate-400">
          <a
            href="#abstract"
            className="hover:text-slate-200 transition-colors"
          >
            Abstract
          </a>
          <a
            href="#contributions"
            className="hover:text-slate-200 transition-colors"
          >
            Contributions
          </a>
          <a href="#methods" className="hover:text-slate-200 transition-colors">
            Methods
          </a>
          <a href="#results" className="hover:text-slate-200 transition-colors">
            Results
          </a>
          <Link
            href="/rubric"
            className="hover:text-slate-200 transition-colors"
          >
            Clinician Rubric
          </Link>
        </nav>
        <Link href="/diagnose">
          <Button
            size="sm"
            className="bg-teal-600 hover:bg-teal-700 text-white text-sm"
          >
            Live Demo <ExternalLink className="ml-1.5 h-3.5 w-3.5" />
          </Button>
        </Link>
      </header>

      <main className="flex-1">
        {/* ── Hero / Title ── */}
        <section className="py-16 px-6 lg:px-12 border-b border-slate-800">
          <div className="max-w-4xl mx-auto text-center space-y-6">
            <p className="text-xs font-semibold tracking-widest uppercase text-teal-400">
              CVPR 2026 Workshop Submission
            </p>
            <h1 className="text-3xl md:text-5xl font-extrabold leading-tight text-slate-100">
              Bridging the Break: A Multi-Agent Framework for
              <br className="hidden md:block" /> Human-Verified Orthopedic
              Diagnosis
            </h1>
            <p className="text-base text-slate-400">Anonymous Authors</p>
            <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
              <Link href="/diagnose">
                <Button className="bg-teal-600 hover:bg-teal-700 text-white gap-2">
                  <ExternalLink className="h-4 w-4" /> Live Demo
                </Button>
              </Link>
              <a
                href="https://github.com/anonymous-submission-research/CVPR-Submission-2026-FRAC-MAS"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Button
                  variant="outline"
                  className="border-slate-600 text-slate-300 hover:bg-slate-800 gap-2"
                >
                  <Github className="h-4 w-4" /> Code
                </Button>
              </a>
              <Link href="/rubric">
                <Button
                  variant="outline"
                  className="border-slate-600 text-slate-300 hover:bg-slate-800 gap-2"
                >
                  <FileText className="h-4 w-4" /> Clinician Rubric
                </Button>
              </Link>
            </div>
          </div>
        </section>

        {/* ── Key Numbers ── */}
        <section className="py-10 px-6 lg:px-12 border-b border-slate-800 bg-slate-900/30">
          <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              value="89.3%"
              label="Ensemble Accuracy"
              sub="8-class HBFMID test set"
            />
            <StatCard
              value="96.4%"
              label="External Detection Rate"
              sub="Roboflow dataset (stacking)"
            />
            <StatCard
              value="4.14 / 5"
              label="Clinical Accuracy Score"
              sub="Mean rating by 3 orthopaedic surgeons"
            />
            <StatCard
              value="68.2%"
              label="Readability ≥ 4/5"
              sub="Layperson comprehensibility"
            />
          </div>
        </section>

        {/* ── Abstract ── */}
        <section
          id="abstract"
          className="py-14 px-6 lg:px-12 border-b border-slate-800"
        >
          <div className="max-w-3xl mx-auto space-y-4">
            <SectionLabel>Abstract</SectionLabel>
            <p className="text-slate-300 leading-relaxed text-base">
              Accurate interpretation of X-ray images is essential for fracture
              diagnosis and management. However, this task remains challenging
              due to complex fracture patterns, variability in clinical
              interpretation, and high case volumes. We trained a stacked
              ensemble of four fine-tuned models (HyperColumn-CBAM DenseNet-169,
              MaxViT, RAD-DINO, and YOLOv26m-cls) alongside four agents:
              Knowledge, Critic, Educational, and Patient Interface, each
              addressing distinct clinical deployment risks — accessibility,
              hallucination, false confidence, and interpretability.
            </p>
            <p className="text-slate-300 leading-relaxed text-base">
              Our results (68.2% of responses scoring ≥ 4 for layperson
              readability) confirm that the multi-agent pipeline effectively
              bridges raw classification outputs and patient-friendly
              explanations. The modest AUC on FracAtlas confirms that features
              from hand and wrist X-rays do not transfer robustly to femoral
              neck or vertebral fractures — bridging this gap will require
              multi-anatomy training data and explicit domain adaptation. The
              system enables patients to better understand their orthopedic
              ailments and assists in identifying fractures in
              resource-constrained settings, acting as a complementary
              diagnostic aid for medical professionals.
            </p>
          </div>
        </section>

        {/* ── Architecture Diagram ── */}
        <section
          id="methods"
          className="py-14 px-6 lg:px-12 border-b border-slate-800 bg-slate-900/30"
        >
          <div className="max-w-5xl mx-auto space-y-6">
            <div>
              <SectionLabel>System Architecture</SectionLabel>
              <SectionTitle>
                End-to-End Multi-Agent Clinical Pipeline
              </SectionTitle>
            </div>
            <div className="border border-slate-700 rounded-lg overflow-hidden bg-white">
              <Image
                src="/figures/architecture-diagram.png"
                alt="MACDSS end-to-end system architecture"
                width={1400}
                height={600}
                className="w-full h-auto"
              />
            </div>
            <FigCaption num="1">
              End-to-end system architecture integrating ensemble inference,
              Grad-CAM localization, multi-agent reasoning, and conformal
              prediction for verifiable clinical report generation. The Patient
              Interface Agent orchestrates downstream routing to the Knowledge
              Agent, Critic Agent, and Educational Agent.
            </FigCaption>
          </div>
        </section>

        {/* ── Contributions ── */}
        <section
          id="contributions"
          className="py-14 px-6 lg:px-12 border-b border-slate-800"
        >
          <div className="max-w-5xl mx-auto space-y-8">
            <div>
              <SectionLabel>Contributions</SectionLabel>
              <SectionTitle>Three Core Technical Contributions</SectionTitle>
            </div>
            <div className="grid md:grid-cols-3 gap-5">
              {[
                {
                  num: "01",
                  title: "Ensemble Inference with Conformal Prediction",
                  desc: "A stacked ensemble of four fine-tuned vision backbones combined via two-pass weighted soft voting. Outputs are wrapped in conformal prediction sets with a distribution-free coverage guarantee, converting point predictions into statistically rigorous differential diagnoses (92.0% empirical coverage at α = 0.10).",
                },
                {
                  num: "02",
                  title: "Multi-Agent Clinical Decision Support",
                  desc: "A four-agent workflow (Patient Interface, Knowledge, Critic, Educational) addressing accessibility, hallucination, false confidence, and interpretability. The Critic Agent performs blind two-step post-hoc verification, flagging ambiguous cases for human review. Post-Critic accuracy improves from 89.3% to 94.6%.",
                },
                {
                  num: "03",
                  title: "External Validation & Clinical Assessment",
                  desc: "Pipeline validated on two external datasets (FracAtlas, Roboflow Bone Break Classification) and assessed by 3 practicing orthopedic surgeons. Substantial inter-rater reliability (Fleiss' κ = 0.72 for accuracy), with 75.0% of responses rated clinically accurate or highly accurate.",
                },
              ].map((c) => (
                <div
                  key={c.num}
                  className="border border-slate-700 bg-slate-900/60 rounded-lg p-6 space-y-3"
                >
                  <p className="text-3xl font-black text-teal-500/30 leading-none">
                    {c.num}
                  </p>
                  <h3 className="text-base font-semibold text-slate-100">
                    {c.title}
                  </h3>
                  <p className="text-sm text-slate-400 leading-relaxed">
                    {c.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ── Model Ensemble ── */}
        <section className="py-14 px-6 lg:px-12 border-b border-slate-800 bg-slate-900/30">
          <div className="max-w-5xl mx-auto space-y-8">
            <div>
              <SectionLabel>Methodology</SectionLabel>
              <SectionTitle>Model Ensemble Design</SectionTitle>
            </div>
            <div className="grid md:grid-cols-2 gap-10 items-start">
              <div className="space-y-4 text-sm text-slate-400 leading-relaxed">
                <p>
                  Fractures vary widely in appearance — from hairline cracks to
                  full structural breaks. Our system employs an ensemble of four
                  architectures to capture this morphological spectrum:
                </p>
                <ul className="space-y-2.5">
                  {[
                    {
                      name: "HyperColumn-CBAM DenseNet-169",
                      desc: "Custom backbone aggregating multi-scale DenseNet-169 features with CBAM channel-spatial attention, preserving both low-level edge detail and high-level structural context.",
                    },
                    {
                      name: "MaxViT",
                      desc: "Multi-axis attention Vision Transformer capturing long-range anatomical dependencies across the full radiograph. Strongest standalone performer at 96.2% accuracy.",
                    },
                    {
                      name: "RAD-DINO",
                      desc: "Microsoft's self-supervised vision backbone pre-trained on radiology images, contributing domain-specific representations that ImageNet-pretrained backbones miss.",
                    },
                    {
                      name: "YOLOv26m-cls",
                      desc: "Adapted for classification, provides a complementary CNN-based inductive bias with efficient feature extraction and strong detection rate on unseen fracture types.",
                    },
                  ].map((m) => (
                    <li key={m.name} className="flex gap-2">
                      <ChevronRight className="h-4 w-4 text-teal-500 flex-shrink-0 mt-0.5" />
                      <span>
                        <span className="font-semibold text-slate-300">
                          {m.name}:{" "}
                        </span>
                        {m.desc}
                      </span>
                    </li>
                  ))}
                </ul>
                <p>
                  Predictions are combined via a two-pass weighted soft voting
                  scheme. A second pass elevates the HyperColumn-CBAM weight for
                  commonly confused categories (Oblique, Transverse and their
                  displaced variants). A stacking meta-learner (logistic
                  regression) achieves the best external detection rate of
                  96.4%.
                </p>
              </div>
              {/* Ablation Table */}
              <div>
                <TableCaption num="1">
                  Single-model ablation on the held-out 8-class test set (no
                  augmentation).
                </TableCaption>
                <div className="overflow-x-auto rounded-lg border border-slate-700">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-800/80 text-slate-300 text-xs uppercase tracking-wider">
                      <tr>
                        <th className="px-4 py-3 text-left">Configuration</th>
                        <th className="px-4 py-3 text-center">Accuracy</th>
                        <th className="px-4 py-3 text-center">F1 (macro)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {[
                        {
                          name: "MaxViT",
                          acc: "96.2%",
                          f1: "0.966",
                          highlight: true,
                        },
                        {
                          name: "HC-CBAM-DenseNet-169",
                          acc: "93.4%",
                          f1: "0.936",
                          highlight: false,
                        },
                        {
                          name: "YOLOv26m-cls",
                          acc: "93.1%",
                          f1: "0.938",
                          highlight: false,
                        },
                        {
                          name: "RAD-DINO",
                          acc: "92.5%",
                          f1: "0.931",
                          highlight: false,
                        },
                        {
                          name: "Ensemble (Stacking) †",
                          acc: "89.3% test",
                          f1: "—",
                          highlight: true,
                        },
                      ].map((row) => (
                        <tr
                          key={row.name}
                          className={
                            row.highlight ? "bg-teal-900/20" : "bg-slate-900/40"
                          }
                        >
                          <td
                            className={`px-4 py-3 ${row.highlight ? "font-semibold text-slate-100" : "text-slate-300"}`}
                          >
                            {row.name}
                          </td>
                          <td className="px-4 py-3 text-center text-slate-300">
                            {row.acc}
                          </td>
                          <td className="px-4 py-3 text-center text-slate-300">
                            {row.f1}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-2 text-xs text-slate-500">
                  † Stacking also achieves 96.4% detection rate on the external
                  Roboflow dataset (140 images, all positive; F1 = 0.982).
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ── Multi-Agent System ── */}
        <section className="py-14 px-6 lg:px-12 border-b border-slate-800">
          <div className="max-w-5xl mx-auto space-y-8">
            <div>
              <SectionLabel>
                MACDSS — Multi-Agent Clinical Decision Support System
              </SectionLabel>
              <SectionTitle>Four-Agent Workflow</SectionTitle>
              <p className="mt-2 text-sm text-slate-400 max-w-2xl">
                Every case is routed through a structured four-agent pipeline
                with clear handoffs, each targeting a specific clinical
                deployment risk.
              </p>
            </div>
            <div className="grid sm:grid-cols-2 gap-4">
              <AgentCard
                number="1"
                name="Patient Interface Agent"
                role="Entry point & orchestrator"
                desc="Provides the chat interface, personalises responses based on patient background, and uses a LangGraph workflow to decide whether to answer immediately or invoke downstream agents for evidence gathering and verification."
              />
              <AgentCard
                number="2"
                name="Knowledge Agent"
                role="Hallucination mitigation via RAG"
                desc="Queries a curated ChromaDB vector store with ICD-10 codes, severity ratings, treatment guidelines, and indexed AO/OTA, Radiopaedia, and FDA AI/ML references. Semantic search via sentence-transformer embeddings; generates only from retrieved material."
              />
              <AgentCard
                number="3"
                name="Critic Agent"
                role="Blind verification & triage"
                desc="Two-step blind-first protocol: (1) independent assessment of the radiograph, then (2) evaluate the ensemble label. Flags cases for human review when it rejects the ensemble, disagrees with confidence ≥ 0.6, or detects statistical ambiguity. Improves confirmed accuracy from 89.3% → 94.6%."
              />
              <AgentCard
                number="4"
                name="Educational Agent"
                role="Patient-friendly translation"
                desc="Uses the Grad-CAM heatmap and verified classification to generate lay summaries with severity and next-steps guidance via Gemini 2.5 Pro. Falls back to template-based generation when API is unavailable. Outputs validated by orthopedic clinicians."
              />
            </div>
          </div>
        </section>

        {/* ── Results ── */}
        <section
          id="results"
          className="py-14 px-6 lg:px-12 border-b border-slate-800 bg-slate-900/30"
        >
          <div className="max-w-5xl mx-auto space-y-12">
            <div>
              <SectionLabel>Results & Analysis</SectionLabel>
              <SectionTitle>Quantitative Evaluation</SectionTitle>
            </div>

            {/* GradCAM row */}
            <div className="space-y-3">
              <h3 className="text-base font-semibold text-slate-200">
                Grad-CAM Attention Analysis
              </h3>
              <div className="border border-slate-700 rounded-lg overflow-hidden bg-slate-950">
                <Image
                  src="/figures/gradcam_single_row.png"
                  alt="Grad-CAM heatmaps comparison for Comminuted fracture"
                  width={1400}
                  height={400}
                  className="w-full h-auto"
                />
              </div>
              <FigCaption num="2">
                Grad-CAM heatmaps for a Comminuted fracture. MaxViT (centre)
                attends to 2.4% of the image, precisely targeting the cortical
                break. HyperColumn-CBAM DenseNet-169 (right) activates 59.8%,
                capturing the surrounding bone and tissue context — a 25×
                difference in spatial coverage reflecting complementary
                diagnostic strategies.
              </FigCaption>
            </div>

            {/* Sample X-ray + GradCAM pair */}
            <div className="space-y-3">
              <h3 className="text-base font-semibold text-slate-200">
                Sample: Oblique Displaced Fracture
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="border border-slate-700 rounded-lg overflow-hidden bg-slate-950">
                  <Image
                    src="/figures/Oblique_Displaced_51.jpg"
                    alt="Original X-ray — Oblique Displaced fracture"
                    width={700}
                    height={700}
                    className="w-full h-auto object-cover"
                  />
                </div>
                <div className="border border-teal-700/40 rounded-lg overflow-hidden bg-slate-950">
                  <Image
                    src="/figures/gradcam_Oblique_Displaced_51_densenet169.png"
                    alt="Grad-CAM heatmap — Oblique Displaced fracture — DenseNet-169"
                    width={700}
                    height={700}
                    className="w-full h-auto object-cover"
                  />
                </div>
              </div>
              <FigCaption num="3">
                Left: Raw X-ray of an Oblique Displaced fracture. Right:
                HyperColumn-CBAM DenseNet-169 Grad-CAM activation map
                highlighting the fracture region and surrounding cortical
                context.
              </FigCaption>
            </div>

            {/* Conformal + Confusion */}
            <div className="space-y-3">
              <h3 className="text-base font-semibold text-slate-200">
                Conformal Coverage &amp; Confusion Matrix
              </h3>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="border border-slate-700 rounded-lg overflow-hidden bg-slate-950">
                    <Image
                      src="/figures/per_class_coverage.png"
                      alt="Per-class accuracy and conformal coverage"
                      width={700}
                      height={500}
                      className="w-full h-auto"
                    />
                  </div>
                  <FigCaption num="4">
                    Per-class accuracy vs. conformal coverage at α = 0.05 and α
                    = 0.10. Coverage gains are largest for the most error-prone
                    classes (Oblique: 70.6% → 82.4%).
                  </FigCaption>
                </div>
                <div className="space-y-2">
                  <div className="border border-slate-700 rounded-lg overflow-hidden bg-slate-950">
                    <Image
                      src="/figures/confusion_matrix.png"
                      alt="Ensemble confusion matrix on 8-class test set"
                      width={700}
                      height={500}
                      className="w-full h-auto"
                    />
                  </div>
                  <FigCaption num="5">
                    Ensemble confusion matrix on the 112-sample held-out test
                    set. Misclassifications concentrate among morphologically
                    adjacent categories: Oblique ↔ Transverse.
                  </FigCaption>
                </div>
              </div>
            </div>

            {/* FracAtlas ROC */}
            <div className="space-y-3">
              <h3 className="text-base font-semibold text-slate-200">
                External Validation — FracAtlas ROC Curves
              </h3>
              <div className="border border-slate-700 rounded-lg overflow-hidden bg-white">
                <Image
                  src="/figures/fracatlas_roc_curves_optimal.png"
                  alt="ROC curves on FracAtlas external dataset"
                  width={1400}
                  height={600}
                  className="w-full h-auto"
                />
              </div>
              <FigCaption num="6">
                ROC curves for base models, ensemble strategies, and triplet
                configurations evaluated on the external FracAtlas dataset (200
                balanced samples). Optimal decision thresholds via Youden&apos;s
                J statistic are shown as solid markers. Best AUC = 0.652,
                reflecting domain shift from hand/wrist to full-body
                musculoskeletal radiographs.
              </FigCaption>
            </div>

            {/* Critic metrics */}
            <div className="space-y-3">
              <h3 className="text-base font-semibold text-slate-200">
                Critic Agent Operational Summary
              </h3>
              <div className="grid md:grid-cols-2 gap-6 items-start">
                <div className="border border-slate-700 rounded-lg overflow-hidden bg-slate-950">
                  <Image
                    src="/figures/critic_metrics.png"
                    alt="Critic agent metrics breakdown"
                    width={700}
                    height={500}
                    className="w-full h-auto"
                  />
                </div>
                <div className="space-y-4">
                  <TableCaption num="2">
                    Critic Agent operational summary on the 112-sample test set
                    (hybrid protocol, t̂ = 0.529).
                  </TableCaption>
                  <div className="overflow-x-auto rounded-lg border border-slate-700">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-800/80 text-slate-300 text-xs uppercase tracking-wider">
                        <tr>
                          <th className="px-4 py-3 text-left">Metric</th>
                          <th className="px-4 py-3 text-right">Value</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800">
                        {[
                          { metric: "Raw ensemble accuracy", value: "89.3%" },
                          {
                            metric: "Confirmed accuracy (post-Critic)",
                            value: "94.6%",
                            highlight: true,
                          },
                          {
                            metric: "Safety margin (confirmed vs. raw)",
                            value: "+5.3 pp",
                            highlight: true,
                          },
                          {
                            metric: "Auto-confirmed / total",
                            value: "92 / 112 (82%)",
                          },
                          {
                            metric: "Flagged uncertain / total",
                            value: "20 / 112 (18%)",
                          },
                        ].map((row) => (
                          <tr
                            key={row.metric}
                            className={
                              row.highlight
                                ? "bg-teal-900/20"
                                : "bg-slate-900/40"
                            }
                          >
                            <td className="px-4 py-2.5 text-slate-300">
                              {row.metric}
                            </td>
                            <td
                              className={`px-4 py-2.5 text-right font-semibold ${row.highlight ? "text-teal-400" : "text-slate-200"}`}
                            >
                              {row.value}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    The Critic concentrates a high-confidence confirmed set
                    (82%) and isolates a compact uncertain cohort (18%) for
                    clinician review, without issuing hard automatic rejections.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ── Human Validation ── */}
        <section className="py-14 px-6 lg:px-12 border-b border-slate-800">
          <div className="max-w-5xl mx-auto space-y-8">
            <div>
              <SectionLabel>Human Validation</SectionLabel>
              <SectionTitle>Clinician Reader Study</SectionTitle>
              <p className="mt-2 text-sm text-slate-400 max-w-2xl">
                Three practicing orthopedic surgeons evaluated system outputs
                using the MACDSS grading protocol. Each rater first provided a
                blind diagnosis, then reviewed the AI output and scored it on a
                5-point Likert scale.
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-8 items-start">
              <div className="overflow-x-auto rounded-lg border border-slate-700">
                <TableCaption num="3">
                  Clinician ratings of educational outputs (1–5 scale) and
                  inter-rater agreement across three orthopaedic raters.
                </TableCaption>
                <table className="w-full text-sm">
                  <thead className="bg-slate-800/80 text-slate-300 text-xs uppercase tracking-wider">
                    <tr>
                      <th className="px-4 py-3 text-left">Dimension</th>
                      <th className="px-4 py-3 text-center">Mean Score</th>
                      <th className="px-4 py-3 text-center">Score ≥ 4 (%)</th>
                      <th className="px-4 py-3 text-center">Fleiss&apos; κ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    <tr className="bg-slate-900/40">
                      <td className="px-4 py-3 text-slate-200 font-medium">
                        Technical Accuracy
                      </td>
                      <td className="px-4 py-3 text-center text-teal-400 font-bold">
                        4.14
                      </td>
                      <td className="px-4 py-3 text-center text-slate-300">
                        75.0%
                      </td>
                      <td className="px-4 py-3 text-center text-slate-300">
                        0.72
                      </td>
                    </tr>
                    <tr className="bg-slate-900/40">
                      <td className="px-4 py-3 text-slate-200 font-medium">
                        Comprehensibility
                      </td>
                      <td className="px-4 py-3 text-center text-teal-400 font-bold">
                        3.95
                      </td>
                      <td className="px-4 py-3 text-center text-slate-300">
                        68.2%
                      </td>
                      <td className="px-4 py-3 text-center text-slate-300">
                        0.65
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="space-y-3 text-sm text-slate-400 leading-relaxed">
                <p>
                  The system demonstrates strong alignment with clinical
                  standards. Fleiss&apos; Kappa indicates{" "}
                  <span className="text-slate-200 font-medium">
                    substantial inter-rater reliability
                  </span>{" "}
                  for technical accuracy (κ = 0.72) and moderate-to-substantial
                  agreement for comprehensibility (κ = 0.65).
                </p>
                <p>
                  A perfect score of 5 was awarded in 43.2% of accuracy
                  evaluations. Edge-case failures surfaced on complex displaced
                  fractures (proximal radius shaft, PIP joint dislocation),
                  reinforcing the necessity of a human-in-the-loop for
                  definitive treatment planning.
                </p>
                <Link href="/rubric">
                  <Button
                    variant="outline"
                    className="mt-2 border-slate-600 text-slate-300 hover:bg-slate-800 gap-2 text-sm"
                  >
                    <FileText className="h-4 w-4" /> View Full Clinician Rubric
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* ── Conformal Prediction Details ── */}
        <section className="py-14 px-6 lg:px-12 border-b border-slate-800 bg-slate-900/30">
          <div className="max-w-5xl mx-auto space-y-6">
            <div>
              <SectionLabel>Statistical Rigor</SectionLabel>
              <SectionTitle>
                Conformal Prediction &amp; Differential Diagnosis
              </SectionTitle>
            </div>
            <div className="grid md:grid-cols-2 gap-8 text-sm text-slate-400 leading-relaxed">
              <div className="space-y-3">
                <p>
                  Rather than a single hard classification, the system wraps
                  ensemble outputs in conformal prediction sets with a
                  distribution-free coverage guarantee using split conformal
                  calibration on the validation set.
                </p>
                <p>
                  At α = 0.10, the procedure achieves{" "}
                  <span className="text-slate-200 font-semibold">
                    92.0% empirical coverage
                  </span>{" "}
                  with an average set size of 1.07. Of 112 predictions, 104
                  (92.9%) are singletons and only 8 are two-element
                  differentials.
                </p>
                <p>
                  This maps naturally to a clinical differential: a set of{" "}
                  {"{Oblique, Oblique Displaced}"} at 90% coverage is more
                  actionable than a potentially wrong single prediction,
                  directly communicating model uncertainty to clinicians.
                </p>
              </div>
              <div className="overflow-x-auto rounded-lg border border-slate-700">
                <TableCaption num="4">
                  Per-class accuracy and conformal coverage on the test set.
                  Coverage failures concentrate among morphologically similar
                  types.
                </TableCaption>
                <table className="w-full text-sm">
                  <thead className="bg-slate-800/80 text-xs uppercase tracking-wider text-slate-300">
                    <tr>
                      <th className="px-3 py-2 text-left">Class</th>
                      <th className="px-3 py-2 text-center">N</th>
                      <th className="px-3 py-2 text-center">Acc.</th>
                      <th className="px-3 py-2 text-center">Cov. α=0.05</th>
                      <th className="px-3 py-2 text-center">Cov. α=0.10</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {[
                      {
                        cls: "Comminuted",
                        n: 17,
                        acc: "100%",
                        c05: "100%",
                        c10: "100%",
                        ok: true,
                      },
                      {
                        cls: "Greenstick",
                        n: 13,
                        acc: "100%",
                        c05: "100%",
                        c10: "100%",
                        ok: true,
                      },
                      {
                        cls: "Healthy",
                        n: 10,
                        acc: "100%",
                        c05: "100%",
                        c10: "100%",
                        ok: true,
                      },
                      {
                        cls: "Oblique",
                        n: 17,
                        acc: "70.6%",
                        c05: "76.5%",
                        c10: "82.4%",
                        ok: false,
                      },
                      {
                        cls: "Oblique Disp.",
                        n: 9,
                        acc: "100%",
                        c05: "100%",
                        c10: "100%",
                        ok: true,
                      },
                      {
                        cls: "Spiral",
                        n: 12,
                        acc: "100%",
                        c05: "100%",
                        c10: "100%",
                        ok: true,
                      },
                      {
                        cls: "Transverse",
                        n: 17,
                        acc: "70.6%",
                        c05: "70.6%",
                        c10: "70.6%",
                        ok: false,
                      },
                      {
                        cls: "Transverse Disp.",
                        n: 17,
                        acc: "88.2%",
                        c05: "88.2%",
                        c10: "94.1%",
                        ok: false,
                      },
                      {
                        cls: "Overall",
                        n: 112,
                        acc: "89.3%",
                        c05: "90.2%",
                        c10: "92.0%",
                        ok: null,
                      },
                    ].map((row) => (
                      <tr
                        key={row.cls}
                        className={
                          row.ok === null
                            ? "bg-teal-900/20 font-semibold"
                            : row.ok
                              ? "bg-slate-900/40"
                              : "bg-amber-900/10"
                        }
                      >
                        <td className="px-3 py-2 text-slate-300">{row.cls}</td>
                        <td className="px-3 py-2 text-center text-slate-400">
                          {row.n}
                        </td>
                        <td
                          className={`px-3 py-2 text-center ${row.ok === false ? "text-amber-400" : "text-slate-300"}`}
                        >
                          {row.acc}
                        </td>
                        <td className="px-3 py-2 text-center text-slate-400">
                          {row.c05}
                        </td>
                        <td
                          className={`px-3 py-2 text-center ${row.ok === null ? "text-teal-400" : "text-slate-300"}`}
                        >
                          {row.c10}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* ── Links ── */}
        <section className="py-14 px-6 lg:px-12 bg-slate-900/30">
          <div className="max-w-5xl mx-auto space-y-6">
            <div className="text-center">
              <SectionLabel>Resources</SectionLabel>
              <SectionTitle>Code, Demo &amp; Protocol</SectionTitle>
            </div>
            <div className="grid sm:grid-cols-3 gap-4">
              <Link
                href="/diagnose"
                className="group border border-slate-700 bg-slate-900/60 hover:border-teal-600/50 hover:bg-teal-950/30 rounded-lg p-6 text-center space-y-2 transition-all"
              >
                <ExternalLink className="h-6 w-6 mx-auto text-teal-400 group-hover:scale-110 transition-transform" />
                <p className="font-semibold text-slate-200">Live Demo</p>
                <p className="text-xs text-slate-500">
                  Interactive fracture diagnosis interface
                </p>
              </Link>
              <a
                href="https://github.com/anonymous-submission-research/CVPR-Submission-2026-FRAC-MAS"
                target="_blank"
                rel="noopener noreferrer"
                className="group border border-slate-700 bg-slate-900/60 hover:border-slate-500 hover:bg-slate-800/60 rounded-lg p-6 text-center space-y-2 transition-all"
              >
                <Github className="h-6 w-6 mx-auto text-slate-400 group-hover:scale-110 transition-transform" />
                <p className="font-semibold text-slate-200">Code Repository</p>
                <p className="text-xs text-slate-500">
                  View anonymous source code
                </p>
              </a>
              <Link
                href="/rubric"
                className="group border border-slate-700 bg-slate-900/60 hover:border-teal-600/50 hover:bg-teal-950/30 rounded-lg p-6 text-center space-y-2 transition-all"
              >
                <FileText className="h-6 w-6 mx-auto text-teal-400 group-hover:scale-110 transition-transform" />
                <p className="font-semibold text-slate-200">Clinician Rubric</p>
                <p className="text-xs text-slate-500">
                  MACDSS evaluation protocol (PDF + web)
                </p>
              </Link>
            </div>
          </div>
        </section>
      </main>

      <footer className="py-6 border-t border-slate-800 text-center text-slate-500 text-xs">
        <p>© 2026 Anonymous Authors · CVPR 2026 Workshop Submission</p>
      </footer>
    </div>
  );
}
