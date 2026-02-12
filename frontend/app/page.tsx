"use client";

import { useState } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/analyze", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setResult(data);
  };

  return (
    <div>
      <h1>Upload your BITalino file</h1>
      <input
        type="file"
        accept=".txt"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />
      <button onClick={handleUpload}>Analyze</button>

      {result && (
        <div>
          <h2>Results:</h2>
          <p>Average anomaly score: {result.level}</p>
          <p>Predictions: {JSON.stringify(result.message)}</p>
        </div>
      )}
    </div>
  );
}
