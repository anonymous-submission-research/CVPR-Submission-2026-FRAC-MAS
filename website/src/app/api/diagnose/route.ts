import { NextRequest, NextResponse } from "next/server";

function _swap_prediction_label(label: string): string {
  if (label === "Transverse") return "Transverse Displaced";
  if (label === "Transverse Displaced") return "Transverse";
  if (label === "Oblique") return "Oblique Displaced";
  if (label === "Oblique Displaced") return "Oblique";
  return label;
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // Forward to Python Backend
    // In production, this URL would be the HF Space URL
    const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:7860";

    const backendFormData = new FormData();
    backendFormData.append("file", file);

    // Forward optional analysis options from the frontend
    const useConformal = formData.get("use_conformal");
    const ensembleMode = formData.get("ensemble_mode");
    const stackerPath = formData.get("stacker_path");

    if (useConformal !== null)
      backendFormData.append("use_conformal", String(useConformal));
    if (ensembleMode !== null)
      backendFormData.append("ensemble_mode", String(ensembleMode));
    if (stackerPath !== null)
      backendFormData.append("stacker_path", String(stackerPath));

    const response = await fetch(`${BACKEND_URL}/diagnose`, {
      method: "POST",
      body: backendFormData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Backend Error:", errorText);
      return NextResponse.json(
        { error: "Failed to process image on backend" },
        { status: 500 },
      );
    }

    const data = await response.json();

    // Check if backend already swapped labels
    const isSwapped =
      data.prediction?.is_label_swapped ||
      data.ensemble?.is_label_swapped ||
      false;

    if (!isSwapped) {
      // Apply swap logic if backend hasn't already
      if (data.prediction) {
        if (data.prediction.ensemble_prediction) {
          data.prediction.ensemble_prediction = _swap_prediction_label(
            data.prediction.ensemble_prediction,
          );
        }
        if (data.prediction.top_class) {
          data.prediction.top_class = _swap_prediction_label(
            data.prediction.top_class,
          );
        }
        if (data.prediction.severity_type) {
          data.prediction.severity_type = _swap_prediction_label(
            data.prediction.severity_type,
          );
        }

        if (data.prediction.all_probabilities) {
          const newProbs: Record<string, number> = {};
          for (const [key, val] of Object.entries(
            data.prediction.all_probabilities as Record<string, any>,
          )) {
            newProbs[_swap_prediction_label(key)] = Number(val);
          }
          data.prediction.all_probabilities = newProbs;
        }

        if (data.prediction.individual_model_predictions) {
          for (const key in data.prediction.individual_model_predictions) {
            const pred = data.prediction.individual_model_predictions[key];
            if (pred.class) {
              pred.class = _swap_prediction_label(pred.class);
            }
          }
        }
      }
    }

    return NextResponse.json(data);
  } catch (error) {
    console.error("Diagnosis Error:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 },
    );
  }
}
