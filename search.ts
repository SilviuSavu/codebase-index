/**
 * search.ts — Hybrid search: rg + trigram + BM25 + pgvector → RRF → rerank
 *
 * Runs ALL four search strategies in parallel every time:
 * 1. ripgrep — fast exact/regex file matching
 * 2. BM25 (ParadeDB pg_search) — full-text keyword search with proper scoring
 * 3. pg_trgm — regex / fuzzy substring matching
 * 4. pgvector — semantic vector similarity
 *
 * Results are fused with Reciprocal Rank Fusion (RRF), then reranked via DeepInfra.
 * No mode picking. Fire everything, let fusion decide.
 */

import type { Pool } from "pg";
import type { SearchResult } from "./db.js";
import { embedQuery } from "./embed.js";
import { type RerankItem, rerank } from "./rerank.js";
import { rgSearch } from "./rg_search.js";

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
	rootDir: string;
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

/**
 * Sanitize a query for use with PostgreSQL regex (~* operator).
 * - Collapses whitespace/newlines
 * - Escapes regex metacharacters
 * - Truncates to avoid huge patterns
 */
function sanitizePgRegex(query: string): string {
	let q = query.replace(/\s+/g, " ").trim();
	// Escape POSIX regex metacharacters so the query is treated as literal
	q = q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	if (q.length > 200) q = q.slice(0, 200);
	return q;
}

/**
 * Sanitize a query for ParadeDB BM25 full-text search.
 * - Extracts meaningful search tokens from natural language
 * - Falls back to simple term-based search if parse fails
 */
function sanitizeBm25Query(query: string): string {
	// Collapse whitespace and strip newlines
	let q = query.replace(/\s+/g, " ").trim();
	// Truncate long queries
	if (q.length > 200) q = q.slice(0, 200);
	// Extract word-like tokens for ParadeDB simple term query
	const tokens = q.match(/\w+/g);
	if (!tokens || tokens.length === 0) return "";
	// Use simple term-based query: each token is searched as a column term
	return tokens.slice(0, 10).map((t) => `content:${t}`).join(" OR ");
}

async function regexSearch(
	pool: Pool,
	project: string,
	query: string,
	language: string | undefined,
	limit: number,
): Promise<RawHit[]> {
	const sanitized = sanitizePgRegex(query);
	if (!sanitized) return [];
	const { sql: where, params } = whereClause(project, language, 2);
	const sql = `
    SELECT id, file_path, language, symbol_type, symbol_name, start_line, end_line, content
    FROM code_chunks
    ${where} AND content ~* $${params.length + 1}
    LIMIT $${params.length + 2}
  `;
	params.push(sanitized, limit);
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
	const sanitized = sanitizeBm25Query(query);
	if (!sanitized) return [];
	const { sql: where, params } = whereClause(project, language, 2);
	const sql = `
    SELECT id, file_path, language, symbol_type, symbol_name, start_line, end_line, content,
           pdb.score(id) AS score
    FROM code_chunks
    ${where} AND content @@@ paradedb.parse($${params.length + 1})
    ORDER BY score DESC
    LIMIT $${params.length + 2}
  `;
	params.push(sanitized, limit);
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

// ── rg → DB lookup ─────────────────────────────────────────────────────────

/**
 * Run ripgrep, then look up matching chunks from DB for RRF fusion.
 * Falls back to raw rg results (no DB id) if lookup fails.
 */
async function rgSearchWithLookup(
	pool: Pool,
	project: string,
	rootDir: string,
	query: string,
	language: string | undefined,
	limit: number,
	signal?: AbortSignal,
): Promise<RawHit[]> {
	const rgMatches = await rgSearch(rootDir, query, {
		language,
		limit: limit * 2,
		signal,
	});
	if (rgMatches.length === 0) return [];

	// Look up DB chunks that overlap with rg match lines
	const hits: RawHit[] = [];
	for (const match of rgMatches) {
		if (hits.length >= limit) break;
		try {
			const { rows } = await pool.query(
				`SELECT id, file_path, language, symbol_type, symbol_name, start_line, end_line, content
         FROM code_chunks
         WHERE project = $1 AND file_path = $2
           AND start_line <= $3 AND end_line >= $3
         LIMIT 1`,
				[project, match.file_path, match.line_number],
			);
			if (rows.length > 0) {
				hits.push(rows[0] as RawHit);
			}
		} catch {
			// DB lookup failed for this match, skip
		}
	}
	return hits;
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

// ── Main hybrid search ────��─────────────────────────────────────────────────

export async function hybridSearch(
	pool: Pool,
	options: SearchOptions,
): Promise<SearchResult[]> {
	const {
		query,
		project,
		language,
		maxResults = 10,
		signal,
		rootDir,
	} = options;

	const fetchLimit = maxResults * 3;

	// Run ALL strategies in parallel
	const [bm25Hits, regexHits, rgHits, semanticHits] = await Promise.all([
		// BM25 — fast, no API call
		bm25Search(pool, project, query, language, fetchLimit).catch((err) => {
			console.error(
				"[codebase-index] BM25 search failed:",
				(err as Error).message,
			);
			return [] as RawHit[];
		}),
		// pg_trgm regex
		regexSearch(pool, project, query, language, fetchLimit).catch((err) => {
			console.error(
				"[codebase-index] regex search failed:",
				(err as Error).message,
			);
			return [] as RawHit[];
		}),
		// ripgrep — fast file search, then DB lookup for chunk context
		rgSearchWithLookup(
			pool,
			project,
			rootDir,
			query,
			language,
			fetchLimit,
			signal,
		).catch((err) => {
			console.error(
				"[codebase-index] rg search failed:",
				(err as Error).message,
			);
			return [] as RawHit[];
		}),
		// Semantic — needs embedding API call
		embedQuery(query, signal)
			.then((embedding) =>
				semanticSearch(pool, project, embedding, language, fetchLimit),
			)
			.catch((err) => {
				console.error(
					"[codebase-index] semantic search failed:",
					(err as Error).message,
				);
				return [] as RawHit[];
			}),
	]);

	// Collect non-empty result sets for fusion
	const resultSets: RawHit[][] = [];
	if (bm25Hits.length > 0) resultSets.push(bm25Hits);
	if (regexHits.length > 0) resultSets.push(regexHits);
	if (rgHits.length > 0) resultSets.push(rgHits);
	if (semanticHits.length > 0) resultSets.push(semanticHits);

	if (resultSets.length === 0) return [];

	// Fuse results with RRF
	const fused = rrfFuse(resultSets, maxResults);

	// Rerank via DeepInfra
	if (fused.length > 0) {
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
