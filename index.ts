/**
 * codebase-index — Pi extension entry point
 *
 * Provides Cursor-like codebase intelligence for local projects.
 * Single `codebase_search` tool registered via pi.registerTool().
 *
 * Architecture:
 *   Workspace files → heuristic chunking → content_hash
 *                                          │
 *                     ┌────────────────────┤
 *                     ▼                    ▼
 *             PostgreSQL (Docker)    OpenRouter API
 *             pg_trgm (regex)        Qwen3-Embedding
 *             tsvector (keyword)
 *             pgvector (semantic)    DeepInfra API
 *             JSONB (meta)           Qwen3-Reranker
 *             content_hash
 */

import { StringEnum } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "typebox";
import {
	closePool,
	ensureSchema,
	getChunkCount,
	getChunksWithoutEmbeddings,
	getPool,
	isHealthy,
	updateEmbeddings,
	upsertChunks,
} from "./db.js";
import { generateEmbeddings } from "./embed.js";
import { type IndexingResult, indexWorkspace } from "./indexer.js";
import { getDeepInfraKey, getOpenRouterKey } from "./keychain.js";
import { hybridSearch, type SearchMode } from "./search.js";
import { type FileWatcher, startWatcher } from "./watcher.js";

// ── State ───────────────────────────────────────────────────────────────────

let currentWatcher: FileWatcher | null = null;
let isIndexing = false;
let poolReady = false;

// ── Helpers ─────────────────────────────────────────────────────────────────

function projectKey(cwd: string): string {
	return Array.from(cwd)
		.map((c) => (/\w/.test(c) ? c : "_"))
		.join("")
		.slice(0, 64);
}

async function ensurePool(): Promise<ReturnType<typeof getPool> | null> {
	if (poolReady) return getPool();
	try {
		const pool = getPool();
		if (await isHealthy(pool)) {
			await ensureSchema(pool);
			poolReady = true;
			return pool;
		}
	} catch {
		// Database not available
	}
	return null;
}

async function runIndexing(
	project: string,
	rootDir: string,
	pool: ReturnType<typeof getPool>,
	signal?: AbortSignal,
): Promise<IndexingResult> {
	const result = await indexWorkspace({
		project,
		rootDir,
		signal,
	});

	if (result.chunks.length > 0) {
		await upsertChunks(pool, result.chunks);
	}

	return result;
}

async function backfillEmbeddings(
	pool: ReturnType<typeof getPool>,
	project: string,
	signal?: AbortSignal,
): Promise<void> {
	const BATCH = 64;
	let processed = 0;

	while (true) {
		if (signal?.aborted) break;
		const chunks = await getChunksWithoutEmbeddings(pool, project, BATCH);
		if (chunks.length === 0) break;

		const texts = chunks.map((c) => c.content);
		try {
			const { embeddings } = await generateEmbeddings(texts, signal);
			const map = new Map<number, number[]>();
			for (let i = 0; i < chunks.length; i++) {
				const chunkId = chunks[i].id;
				if (chunkId !== undefined && embeddings[i]) {
					map.set(chunkId, embeddings[i]);
				}
			}
			await updateEmbeddings(pool, map);
			processed += map.size;
		} catch (err) {
			console.error(
				"[codebase-index] embedding backfill batch failed:",
				(err as Error).message,
			);
			break;
		}
	}

	if (processed > 0) {
		console.log(`[codebase-index] backfilled ${processed} embeddings`);
	}
}

// ── Status display helper ───────────────────────────────────────────────────

interface StatusInfo {
	hasOpenRouter: boolean;
	hasDeepInfra: boolean;
	chunkCount: number;
	embeddedCount: number;
	project: string;
}

function formatStatus(info: StatusInfo): string {
	const lines = [
		`Project: ${info.project}`,
		`Total chunks: ${info.chunkCount}`,
		`With embeddings: ${info.embeddedCount} / ${info.chunkCount}`,
	];
	if (info.hasOpenRouter) {
		lines.push("Semantic search: OPENROUTER_API_KEY set");
	} else {
		lines.push("Semantic search: OPENROUTER_API_KEY missing");
	}
	if (info.hasDeepInfra) {
		lines.push("Reranking: DEEPINFRA_API_KEY set");
	} else {
		lines.push("Reranking: DEEPINFRA_API_KEY missing");
	}
	return lines.join("\n");
}

// ── Extension ───────────────────────────────────────────────────────────────

export default function codebaseIndexExtension(pi: ExtensionAPI) {
	pi.on("session_start", async (_event, ctx) => {
		const project = projectKey(ctx.cwd);
		const pool = await ensurePool();
		if (!pool) {
			ctx.ui.notify(
				"codebase-index: PostgreSQL not available. Run `docker compose up -d` in ~/.pi/agent/extensions/codebase-index/",
				"warning",
			);
			return;
		}

		const chunkCount = await getChunkCount(pool, project);
		if (chunkCount === 0) {
			ctx.ui.notify(
				"codebase-index: Indexing workspace for the first time...",
				"info",
			);
		} else {
			ctx.ui.notify(`codebase-index: ${chunkCount} chunks indexed`, "info");
		}

		// Start indexing in background (non-blocking)
		// ctx.signal is only available during agent turns, not session_start
		if (!isIndexing) {
			isIndexing = true;
			runIndexing(project, ctx.cwd, pool)
				.then(async (result) => {
					isIndexing = false;
					if (result.totalChunks > 0) {
						ctx.ui.notify(
							`codebase-index: Indexed ${result.totalFiles} files → ${result.totalChunks} chunks (${result.errors.length} errors)`,
							result.errors.length > 0 ? "warning" : "info",
						);
					}
					// Backfill embeddings in background
					await backfillEmbeddings(pool, project);
				})
				.catch((err) => {
					isIndexing = false;
					console.error("[codebase-index] indexing error:", err);
				});
		}

		// Start file watcher
		if (!currentWatcher) {
			currentWatcher = await startWatcher({
				pool,
				project,
				rootDir: ctx.cwd,
				onReindex(file, chunks) {
					console.log(`[codebase-index] re-indexed ${file} (${chunks} chunks)`);
				},
			});
		}
	});

	pi.on("session_shutdown", async () => {
		if (currentWatcher) {
			await currentWatcher.stop();
			currentWatcher = null;
		}
		await closePool();
		poolReady = false;
	});

	// Register the codebase_search tool
	pi.registerTool({
		name: "codebase_search",
		label: "Codebase Search",
		description:
			"Search the local codebase using regex, keyword, or semantic search. " +
			"Returns matching code chunks with file paths, line numbers, and symbol info. " +
			"Use mode='regex' for pattern matching, 'keyword' for identifier search, " +
			"'semantic' for natural language queries, or 'auto' to let the system decide.",
		promptSnippet:
			"Search local codebase by regex, keyword, or natural language",
		promptGuidelines: [
			"Use codebase_search for finding code in the LOCAL project workspace.",
			"Use code_search for searching the INTERNET for code examples and docs.",
			"codebase_search supports regex patterns, keyword lookups, and semantic natural language queries.",
			"For 'where is X defined?' or 'how is Y used?', prefer codebase_search with mode='auto'.",
		],
		parameters: Type.Object({
			query: Type.String({
				description:
					"Search query. Can be a regex pattern, keyword/identifier, or natural language description.",
			}),
			mode: Type.Optional(
				StringEnum(["auto", "regex", "keyword", "semantic"] as const, {
					description:
						"Search mode: 'auto' detects best strategy, 'regex' for pattern matching, " +
						"'keyword' for full-text search, 'semantic' for meaning-based search. Default: auto.",
					default: "auto",
				}),
			),
			language: Type.Optional(
				Type.String({
					description:
						"Filter results by programming language (e.g. 'typescript', 'python').",
				}),
			),
			maxResults: Type.Optional(
				Type.Number({
					description: "Maximum number of results to return. Default: 10.",
					default: 10,
					minimum: 1,
					maximum: 50,
				}),
			),
		}),

		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			const pool = await ensurePool();
			if (!pool) {
				return {
					content: [
						{
							type: "text" as const,
							text:
								"codebase-index: PostgreSQL database is not available. " +
								"Please start it with: docker compose -f ~/.pi/agent/extensions/codebase-index/docker-compose.yml up -d",
						},
					],
					details: {},
					isError: true,
				};
			}

			const project = projectKey(ctx.cwd);
			const chunkCount = await getChunkCount(pool, project);
			if (chunkCount === 0) {
				return {
					content: [
						{
							type: "text" as const,
							text:
								"codebase-index: No chunks indexed for this project yet. " +
								"Indexing may still be in progress. Try again in a moment.",
						},
					],
					details: { chunkCount: 0, project },
				};
			}

			try {
				const results = await hybridSearch(pool, {
					query: params.query,
					mode: (params.mode as SearchMode) || "auto",
					project,
					language: params.language,
					maxResults: params.maxResults ?? 10,
					signal,
				});

				if (results.length === 0) {
					return {
						content: [
							{
								type: "text" as const,
								text:
									`No results found for "${params.query}". ` +
									`The codebase has ${chunkCount} indexed chunks. Try a different query or mode.`,
							},
						],
						details: { query: params.query, mode: params.mode, chunkCount },
					};
				}

				// Format results for the LLM
				const formatted = results
					.map((r, i) => {
						const header = [
							`--- Result ${i + 1} (score: ${r.score.toFixed(4)}) ---`,
							`File: ${r.file_path}:${r.start_line}-${r.end_line}`,
						];
						if (r.language) header.push(`Language: ${r.language}`);
						if (r.symbol_type && r.symbol_name) {
							header.push(`Symbol: ${r.symbol_type} ${r.symbol_name}`);
						}
						header.push("", r.content);
						return header.join("\n");
					})
					.join("\n\n");

				const resultMeta = results.map((r) => ({
					file_path: r.file_path,
					start_line: r.start_line,
					end_line: r.end_line,
					symbol_type: r.symbol_type,
					symbol_name: r.symbol_name,
					score: r.score,
				}));

				return {
					content: [{ type: "text" as const, text: formatted }],
					details: {
						query: params.query,
						mode: params.mode || "auto",
						resultCount: results.length,
						chunkCount,
						results: resultMeta,
					},
				};
			} catch (err) {
				return {
					content: [
						{
							type: "text" as const,
							text: `codebase-index search error: ${(err as Error).message}`,
						},
					],
					details: {},
					isError: true,
				};
			}
		},
	});

	// Register a command to manually trigger re-indexing
	pi.registerCommand("reindex", {
		description: "Re-index the codebase from scratch",
		handler: async (_args, ctx) => {
			const pool = await ensurePool();
			if (!pool) {
				ctx.ui.notify("codebase-index: Database not available", "error");
				return;
			}

			const project = projectKey(ctx.cwd);
			ctx.ui.notify("codebase-index: Re-indexing workspace...", "info");

			try {
				const result = await runIndexing(project, ctx.cwd, pool);
				ctx.ui.notify(
					`codebase-index: Re-indexed ${result.totalFiles} files → ${result.totalChunks} chunks`,
					result.errors.length > 0 ? "warning" : "info",
				);
				// Backfill embeddings
				await backfillEmbeddings(pool, project);
				ctx.ui.notify("codebase-index: Embeddings up to date", "info");
			} catch (err) {
				ctx.ui.notify(
					`codebase-index: Error: ${(err as Error).message}`,
					"error",
				);
			}
		},
	});

	// Register a command to check index status
	pi.registerCommand("index-status", {
		description: "Show codebase index statistics",
		handler: async (_args, ctx) => {
			const pool = await ensurePool();
			if (!pool) {
				ctx.ui.notify("codebase-index: Database not available", "error");
				return;
			}

			const project = projectKey(ctx.cwd);
			const chunkCount = await getChunkCount(pool, project);
			const { rows } = await pool.query(
				"SELECT count(*) as cnt FROM code_chunks WHERE project = $1 AND embedding IS NOT NULL",
				[project],
			);
			const embeddedCount = Number(rows[0]?.cnt ?? 0);

			ctx.ui.notify(
				formatStatus({
					project,
					chunkCount,
					embeddedCount,
					hasOpenRouter: !!getOpenRouterKey(),
					hasDeepInfra: !!getDeepInfraKey(),
				}),
				"info",
			);
		},
	});
}
