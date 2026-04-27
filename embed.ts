/**
 * embed.ts — Embedding generation via DeepInfra API
 *
 * Uses Qwen3-Embedding-8B (4096d, 32K context) through DeepInfra.
 * Batches embedding requests for efficiency.
 */

import { getDeepInfraKey } from "./keychain.js";

const DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/embeddings";
const MODEL = "Qwen/Qwen3-Embedding-8B";
const EMBEDDING_DIM = 1024;
const BATCH_SIZE = 64;

export interface EmbeddingResponse {
  embeddings: number[][];
  tokens: number;
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

async function embedBatch(
  texts: string[],
  signal?: AbortSignal
): Promise<EmbeddingResponse> {
  const resp = await fetch(DEEPINFRA_URL, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${await getApiKey()}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL,
      input: texts,
      dimensions: EMBEDDING_DIM,
    }),
    signal,
  });

  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(
      `DeepInfra embedding API error ${resp.status}: ${body.slice(0, 200)}`
    );
  }

  const data = (await resp.json()) as {
    data: Array<{ embedding: number[] }>;
    usage?: { total_tokens: number };
  };

  const embeddings = data.data.map((d) => d.embedding);
  const tokens = data.usage?.total_tokens ?? 0;

  return { embeddings, tokens };
}

/**
 * Generate embeddings for an array of text strings.
 * Automatically batches requests to stay within API limits.
 */
export async function generateEmbeddings(
  texts: string[],
  signal?: AbortSignal
): Promise<EmbeddingResponse> {
  if (texts.length === 0) {
    return { embeddings: [], tokens: 0 };
  }

  if (texts.length <= BATCH_SIZE) {
    return embedBatch(texts, signal);
  }

  const allEmbeddings: number[][] = [];
  let totalTokens = 0;

  for (let i = 0; i < texts.length; i += BATCH_SIZE) {
    if (signal?.aborted) break;
    const batch = texts.slice(i, i + BATCH_SIZE);
    const result = await embedBatch(batch, signal);
    allEmbeddings.push(...result.embeddings);
    totalTokens += result.tokens;
  }

  return { embeddings: allEmbeddings, tokens: totalTokens };
}

/**
 * Generate a single embedding for a query string.
 * Prefixes with "query: " for retrieval-optimized embeddings.
 */
export async function embedQuery(
  query: string,
  signal?: AbortSignal
): Promise<number[]> {
  const { embeddings } = await embedBatch([`query: ${query}`], signal);
  return embeddings[0];
}
