/**
 * search.ts — Hybrid search: trigram + BM25 + pgvector → RRF → rerank
 *
 * Combines three search strategies via Reciprocal Rank Fusion (RRF):
 * 1. pg_trgm — regex / fuzzy substring matching
 * 2. BM25 (ParadeDB pg_search) — full-text keyword search with proper scoring
 * 3. pgvector — semantic vector similarity
 *
 * The results are fused with RRF, then optionally reranked via DeepInfra.
 */

import type { Pool } from "pg";
import type { SearchResult } from "./db.js";
import { embedQuery } from "./embed.js";
import { type RerankItem, rerank } from "./rerank.js";

// RRF constant (standard value from academic literature)
const RRF_K = 60;

export type SearchMode = "auto" | "regex" | "keyword" | "semantic";

interface SearchOptions {
	query: string;
	mode: SearchMode;
	project: string;
	language?: string;
	maxResults?: number;
	signal?: AbortSignal;
}

/**
 * Detect the best search mode for a query.
 */
function detectMode(query: string): SearchMode {
	if (/[\\^$.*+?[\]{}()|]/.test(query)) return "regex";
	const words = query.split(/\s+/);
	if (words.length <= 3 && words.every((w) => /^[\w.]+$/.test(w))) {
		return "keyword";
	}
	return "semantic";
}

// ── Individual search strategies ────────────────────────────────────────────

interface RawHit {
	id: number;
	file_path: string;
	language: string | null;
	symbol_type: string | null;
	symbol_name: string | null;
	start_line: number | null;
	end_line: number | null;
	content: string;
}

function whereClause(
	project: string,
	language: string | undefined,
	paramOffset: number,
): {
	sql: string;
	params: unknown[];
} {
	const params: unknown[] = [project];
	let sql = `WHERE project = $1`;
	if (language) {
		sql += ` AND language = $${paramOffset}`;
		params.push(language);
	}
	return { sql, params };
}

async function regexSearch(
	pool: Pool,
	project: string,
	query: string,
	language: string | undefined,
	limit: number,
): Promise<RawHit[]> {
	const { sql: where, params } = whereClause(project, language, 2);
	const sql = `
    SELECT id, file_path, language, symbol_type, symbol_name, start_line, end_line, content
    FROM code_chunks
    ${where} AND content ~* $${params.length + 1}
    LIMIT $${params.length + 2}
  `;
	params.push(query, limit);
	const { rows } = await pool.query(sql, params);
	return rows as RawHit[];
}

async function bm25Search(
	pool: Pool,
	project: string,
	query: string,
	language: string | undefined,
	limit: number,
): Promise<RawHit[]> {
	// ParadeDB pg_search: uses @@@ operator and pdb.score() for BM25 scoring
	const { sql: where, params } = whereClause(project, language, 2);
	const sql = `
    SELECT id, file_path, language, symbol_type, symbol_name, start_line, end_line, content,
           pdb.score(id) AS score
    FROM code_chunks
    ${where} AND content @@@ paradedb.parse($${params.length + 1})
    ORDER BY score DESC
    LIMIT $${params.length + 2}
  `;
	params.push(query, limit);
	const { rows } = await pool.query(sql, params);
	return rows as RawHit[];
}

async function semanticSearch(
	pool: Pool,
	project: string,
	queryEmbedding: number[],
	language: string | undefined,
	limit: number,
): Promise<RawHit[]> {
	const vecStr = `[${queryEmbedding.join(",")}]`;
	const { sql: where, params } = whereClause(project, language, 2);
	const sql = `
    SELECT id, file_path, language, symbol_type, symbol_name, start_line, end_line, content,
           1 - (embedding <=> $${params.length + 1}::vector) AS similarity
    FROM code_chunks
    ${where} AND embedding IS NOT NULL
    ORDER BY embedding <=> $${params.length + 1}::vector
    LIMIT $${params.length + 2}
  `;
	params.push(vecStr, limit);
	const { rows } = await pool.query(sql, params);
	return rows as RawHit[];
}

// ── RRF Fusion ──────────────────────────────────────────────────────────────

function rrfFuse(resultSets: RawHit[][], maxResults: number): SearchResult[] {
	const idRanks = new Map<number, number[]>();

	for (const hits of resultSets) {
		hits.forEach((hit, idx) => {
			const ranks = idRanks.get(hit.id);
			if (ranks) {
				ranks.push(idx + 1);
			} else {
				idRanks.set(hit.id, [idx + 1]);
			}
		});
	}

	const hitMap = new Map<number, RawHit>();
	for (const hits of resultSets) {
		for (const hit of hits) {
			if (!hitMap.has(hit.id)) {
				hitMap.set(hit.id, hit);
			}
		}
	}

	const scored: SearchResult[] = [];
	for (const [id, ranks] of idRanks) {
		const hit = hitMap.get(id);
		if (!hit) continue;
		const score = ranks.reduce((sum, rank) => sum + 1 / (RRF_K + rank), 0);
		scored.push({
			id,
			file_path: hit.file_path,
			language: hit.language,
			symbol_type: hit.symbol_type,
			symbol_name: hit.symbol_name,
			start_line: hit.start_line,
			end_line: hit.end_line,
			content: hit.content,
			score,
		});
	}

	scored.sort((a, b) => b.score - a.score);
	return scored.slice(0, maxResults);
}

// ── Main hybrid search ──────────────────────────────────────────────────────

export async function hybridSearch(
	pool: Pool,
	options: SearchOptions,
): Promise<SearchResult[]> {
	const {
		query,
		mode: rawMode,
		project,
		language,
		maxResults = 10,
		signal,
	} = options;

	const mode = rawMode === "auto" ? detectMode(query) : rawMode;
	const fetchLimit = maxResults * 3;

	const resultSets: RawHit[][] = [];

	// Always include BM25 keyword search (fast, no API call)
	try {
		const bm25Hits = await bm25Search(
			pool,
			project,
			query,
			language,
			fetchLimit,
		);
		resultSets.push(bm25Hits);
	} catch (err) {
		console.error(
			"[codebase-index] BM25 search failed:",
			(err as Error).message,
		);
	}

	// Regex search for regex/keyword modes
	if (mode === "regex" || mode === "keyword") {
		try {
			const regexHits = await regexSearch(
				pool,
				project,
				query,
				language,
				fetchLimit,
			);
			resultSets.push(regexHits);
		} catch (err) {
			console.error(
				"[codebase-index] regex search failed:",
				(err as Error).message,
			);
		}
	}

	// Semantic search for semantic/auto modes
	if (mode === "semantic" || mode === "auto") {
		try {
			const queryEmbedding = await embedQuery(query, signal);
			const semanticHits = await semanticSearch(
				pool,
				project,
				queryEmbedding,
				language,
				fetchLimit,
			);
			resultSets.push(semanticHits);
		} catch (err) {
			console.error(
				"[codebase-index] semantic search failed:",
				(err as Error).message,
			);
		}
	}

	// Fuse results with RRF
	const fused = rrfFuse(resultSets, maxResults);

	// Rerank if we have semantic or auto mode and results
	if ((mode === "semantic" || mode === "auto") && fused.length > 0) {
		try {
			const rerankItems: RerankItem[] = fused.map((r) => ({
				id: r.id,
				text: r.content.slice(0, 2000),
			}));

			const reranked = await rerank(query, rerankItems, {
				signal,
				topN: maxResults,
			});

			const resultMap = new Map(fused.map((r) => [r.id, r]));
			return reranked
				.map((rr) => {
					const result = resultMap.get(rr.id);
					if (!result) return null;
					return { ...result, score: rr.score };
				})
				.filter((r): r is SearchResult => r !== null);
		} catch (err) {
			console.error(
				"[codebase-index] rerank failed, using RRF scores:",
				(err as Error).message,
			);
		}
	}

	return fused;
}
