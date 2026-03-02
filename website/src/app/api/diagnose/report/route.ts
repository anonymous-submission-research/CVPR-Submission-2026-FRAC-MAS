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
    // Forward the original multipart body and Content-Type header
    const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:7860";
    const rawBody = await req.arrayBuffer();
    const contentType =
      req.headers.get("content-type") || "multipart/form-data";

    // Try configured BACKEND_URL first, fall back to localhost (dev) if it fails.
    let response = await fetch(`${BACKEND_URL}/diagnose/report`, {
      method: "POST",
      body: rawBody,
      headers: {
        "content-type": contentType,
      },
    });
    if (!response.ok) {
      console.warn(
        `Primary backend ${BACKEND_URL} returned ${response.status}; trying http://127.0.0.1:7860`,
      );
      try {
        response = await fetch(`http://127.0.0.1:7860/diagnose/report`, {
          method: "POST",
          body: rawBody,
          headers: {
            "content-type": contentType,
          },
        });
      } catch (e) {
        console.error("Fallback to local backend failed", e);
      }
    }

    if (!response.ok) {
      const text = await response.text();
      console.error("Backend report error:", text);
      return NextResponse.json(
        { error: "Backend report failed" },
        { status: 500 },
      );
    }

    // Intercept JSON responses to ensure label swapping consistency
    const respContentType = response.headers.get("content-type") || "";
    if (respContentType.includes("application/json")) {
      try {
        const data = await response.json();

        // Allow backend to signal if it already swapped labels
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
                data.prediction.all_probabilities,
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
      } catch (e) {
        console.error("Error parsing/processing JSON response from backend", e);
        // Fallback to streaming original response if parsing fails
        // Note: response.json() consumes the body, so we can't re-read it easily from 'response' object if it was partially read.
        // But if json() fails, it likely wasn't valid JSON or empty.
        // We should probably clone the response before reading json if we wanted to fallback,
        // but here we only try json() if content-type is json.
      }
    }

    // Stream the response back to the client
    const arrayBuffer = await response.arrayBuffer();
    const headers: Record<string, string> = {
      "Content-Type":
        response.headers.get("content-type") || "application/octet-stream",
    };
    const disposition = response.headers.get("content-disposition");
    if (disposition) headers["Content-Disposition"] = disposition;

    return new NextResponse(arrayBuffer, { status: 200, headers });
  } catch (err) {
    console.error(err);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 },
    );
  }
}
