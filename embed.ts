/**
 * embed.ts — Embedding generation via OpenRouter API
 *
 * Uses Qwen3-Embedding-0.6B (1024d, 32K context) through OpenRouter.
 * Batches embedding requests for efficiency.
 */

import { getOpenRouterKey } from "./keychain.js";

const OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings";
const MODEL = "qwen/qwen3-embedding";
const EMBEDDING_DIM = 1024;
const BATCH_SIZE = 64;

export interface EmbeddingResponse {
  embeddings: number[][];
  tokens: number;
}

async function getApiKey(): Promise<string> {
  const key = getOpenRouterKey();
  if (!key) {
    throw new Error(
      "OPENROUTER_API_KEY not found in env or Keychain. " +
      "Set it with: export OPENROUTER_API_KEY=... or " +
      "security add-generic-password -a openrouter -s OPENROUTER_API_KEY -w <key>"
    );
  }
  return key;
}

async function embedBatch(
  texts: string[],
  signal?: AbortSignal
): Promise<EmbeddingResponse> {
  const resp = await fetch(OPENROUTER_URL, {
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
      `OpenRouter embedding API error ${resp.status}: ${body.slice(0, 200)}`
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
