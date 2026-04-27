/**
 * rerank.ts — Reranking via DeepInfra API
 *
 * Uses Qwen3-Reranker-0.6B to re-score search results for relevance.
 */

import { getDeepInfraKey } from "./keychain.js";

const DEEPINFRA_URL = "https://api.deepinfra.com/v1/rerank";
const MODEL = "Qwen/Qwen3-Reranker-0.6B";
const MAX_RERANK_DOCS = 20;

export interface RerankItem {
  id: number;
  text: string;
}

export interface RerankResult {
  id: number;
  score: number;
}

async function getApiKey(): Promise<string> {
  const key = getDeepInfraKey();
  if (!key) {
    throw new Error(
      "DEEPINFRA_API_KEY not found in env or Keychain. " +
      "Set it with: export DEEPINFRA_API_KEY=... or " +
      "security add-generic-password -a deepinfra -s DEEPINFRA_API_KEY -w <key>"
    );
  }
  return key;
}

/**
 * Rerank documents against a query. Returns items sorted by relevance score.
 */
export async function rerank(
  query: string,
  documents: RerankItem[],
  options?: { signal?: AbortSignal; topN?: number }
): Promise<RerankResult[]> {
  if (documents.length === 0) return [];

  const topN = options?.topN ?? 10;
  const inputDocs = documents.slice(0, MAX_RERANK_DOCS);

  const resp = await fetch(DEEPINFRA_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${await getApiKey()}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      query,
      documents: inputDocs.map((d) => d.text),
      top_n: Math.min(topN, inputDocs.length),
    }),
    signal: options?.signal,
  });

  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(
      `DeepInfra rerank API error ${resp.status}: ${body.slice(0, 200)}`
    );
  }

  const data = (await resp.json()) as {
    results: Array<{ index: number; relevance_score: number }>;
  };

  return data.results.map((r) => ({
    id: inputDocs[r.index].id,
    score: r.relevance_score,
  }));
}
