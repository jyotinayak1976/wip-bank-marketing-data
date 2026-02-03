import { useState } from "react";
import "./styles.css";

const API_URL = "http://localhost:8000/predict";

const initialFormState = {
  age: 35,
  job: "admin.",
  marital: "married",
  education: "tertiary",
  default: "no",
  balance: 1500,
  housing: "yes",
  loan: "no",
  contact: "cellular",
  day: 5,
  month: "may",
  campaign: 2,
  pdays: 999,
  previous: 0,
  poutcome: "unknown"
};

const fieldMeta = [
  { name: "age", label: "Age", type: "number" },
  { name: "job", label: "Job", type: "text" },
  { name: "marital", label: "Marital Status", type: "text" },
  { name: "education", label: "Education", type: "text" },
  { name: "default", label: "Default", type: "text" },
  { name: "balance", label: "Balance", type: "number" },
  { name: "housing", label: "Housing", type: "text" },
  { name: "loan", label: "Loan", type: "text" },
  { name: "contact", label: "Contact", type: "text" },
  { name: "day", label: "Day", type: "number" },
  { name: "month", label: "Month", type: "text" },
  { name: "campaign", label: "Campaign", type: "number" },
  { name: "pdays", label: "Pdays", type: "number" },
  { name: "previous", label: "Previous", type: "number" },
  { name: "poutcome", label: "Poutcome", type: "text" }
];

export default function App() {
  const [formState, setFormState] = useState(initialFormState);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (event) => {
    const { name, value, type } = event.target;
    setFormState((prev) => ({
      ...prev,
      [name]: type === "number" ? Number(value) : value
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: formState })
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || "Prediction failed");
      }

      const payload = await response.json();
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <header>
        <h1>Bank Marketing Predictor</h1>
        <p>Submit a customer profile to estimate subscription likelihood.</p>
      </header>

      <form className="card" onSubmit={handleSubmit}>
        <div className="grid">
          {fieldMeta.map((field) => (
            <label key={field.name}>
              <span>{field.label}</span>
              <input
                name={field.name}
                type={field.type}
                value={formState[field.name]}
                onChange={handleChange}
                step={field.type === "number" ? "1" : undefined}
              />
            </label>
          ))}
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Scoring..." : "Get Prediction"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section className="result">
          <h2>Prediction</h2>
          <p>
            Outcome: <strong>{result.prediction === 1 ? "Yes" : "No"}</strong>
          </p>
          <p>Probability yes: {result.probability_yes.toFixed(3)}</p>
          <p>Probability no: {result.probability_no.toFixed(3)}</p>
        </section>
      )}
    </div>
  );
}
