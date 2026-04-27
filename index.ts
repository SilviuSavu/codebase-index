/**
 * codebase-index — Pi extension entry point
 *
 * Provides Cursor-like codebase intelligence for local projects.
 * Single `codebase_search` tool registered via pi.registerTool().
 */

import { StringEnum } from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import {
	closePool,
	ensureSchema,
	getChunkCount,
	getChunksWithoutEmbeddings,
	getPool,
	isHealthy,
	updateEmbeddings,
	upsertChunks,
	upsertFileHash,
} from "./db.js";
import { extractGraph } from "./edge_extractor.js";
import { generateEmbeddings } from "./embed.js";
import {
	closeGraphPool,
	ensureGraphSchema,
	expandContext,
	getGraphPool,
	getGraphStats,
	isGraphHealthy as isGraphDbHealthy,
	upsertEdges,
	upsertNodes,
} from "./graph.js";
import { type IndexingResult, indexWorkspace } from "./indexer.js";
import { getDeepInfraKey } from "./keychain.js";
import { hybridSearch, type SearchMode } from "./search.js";
import { type FileWatcher, startWatcher } from "./watcher.js";

// ── State ───────────────────────────────────────────────────────────────────

let currentWatcher: FileWatcher | null = null;
let isIndexing = false;
let poolReady = false;
let graphPoolReady = false;
let backgroundTask: Promise<void> | null = null;
let backgroundAbort: AbortController | null = null;

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

async function ensureGraphPool(): Promise<ReturnType<
	typeof getGraphPool
> | null> {
	if (graphPoolReady) return getGraphPool();
	try {
		const pool = getGraphPool();
		if (await isGraphDbHealthy(pool)) {
			await ensureGraphSchema(pool);
			graphPoolReady = true;
			return pool;
		}
	} catch {
		// Graph database not available
	}
	return null;
}

async function runIndexing(
	project: string,
	rootDir: string,
	pool: ReturnType<typeof getPool>,
	signal?: AbortSignal,
): Promise<IndexingResult> {
	const result = await indexWorkspace({ project, rootDir, signal, pool });

	if (result.chunks.length > 0) {
		await upsertChunks(pool, result.chunks);
		const byFile = new Map<string, string>();
		for (const chunk of result.chunks) {
			const key = `${chunk.project}\0${chunk.file_path}`;
			if (!byFile.has(key)) byFile.set(key, chunk.content_hash);
		}
		for (const [key, hash] of byFile) {
			const sepIdx = key.indexOf("\0");
			await upsertFileHash(
				pool,
				key.slice(0, sepIdx),
				key.slice(sepIdx + 1),
				hash,
			);
		}

		// Build knowledge graph from indexed files
		await buildGraph(rootDir, result);
	}
	return result;
}

async function buildGraph(
	_result: unknown,
	result: IndexingResult,
): Promise<void> {
	const graphPool = await ensureGraphPool();
	if (!graphPool) return;

	try {
		// Group chunks by file for graph extraction
		const byFile = new Map<string, typeof result.chunks>();
		for (const chunk of result.chunks) {
			const list = byFile.get(chunk.file_path) ?? [];
			list.push(chunk);
			byFile.set(chunk.file_path, list);
		}

		const allNodes: Array<import("./graph.js").GraphNode> = [];
		const allEdges: Array<import("./graph.js").EdgeSpec> = [];

		for (const [filePath, chunks] of byFile) {
			// Use content from first chunk (represents the file)
			const content = chunks.map((c) => c.content).join("\n");
			const language = chunks[0]?.language ?? "unknown";
			const graph = extractGraph(content, filePath, language);
			allNodes.push(...graph.nodes);
			allEdges.push(...graph.edges);
		}

		if (allNodes.length > 0) {
			await upsertNodes(graphPool, allNodes);
		}
		if (allEdges.length > 0) {
			const edgeCount = await upsertEdges(graphPool, allEdges);
			if (edgeCount > 0) {
				console.log(
					`[codebase-index] graph: ${allNodes.length} nodes, ${edgeCount} edges`,
				);
			}
		}
	} catch (err) {
		console.error(
			"[codebase-index] graph build error:",
			(err as Error).message,
		);
	}
}

async function buildGraphContext(
	results: Array<{
		file_path: string;
		symbol_name: string | null;
		symbol_type: string | null;
	}>,
): Promise<string> {
	try {
		const graphPool = await ensureGraphPool();
		if (!graphPool) return "";
		const expanded = await expandContext(graphPool, results, 5);
		if (expanded.length === 0) return "";
		const lines = expanded
			.map(
				(e) =>
					`  ${e.file_path}: ${e.symbol_type} ${e.symbol_name} (${e.relation})`,
			)
			.join("\n");
		return `\n[graph] Related symbols:\n${lines}`;
	} catch {
		return "";
	}
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

// Question-word detection via Set (avoids high-complexity regex alternation)
const QUESTION_WORDS = new Set([
	"how", "what", "why", "when", "where", "who",
	"is", "are", "do", "does", "can", "should", "will", "would",
]);

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

		if (!isIndexing) {
			isIndexing = true;
			backgroundAbort = new AbortController();
			const signal = backgroundAbort.signal;

			backgroundTask = runIndexing(project, ctx.cwd, pool, signal)
				.then(async (result) => {
					if (signal.aborted) {
						isIndexing = false;
						return;
					}
					isIndexing = false;
					if (result.totalChunks > 0) {
						ctx.ui.notify(
							`codebase-index: Indexed ${result.totalFiles} files → ${result.totalChunks} chunks (${result.errors.length} errors)`,
							result.errors.length > 0 ? "warning" : "info",
						);
					}
					await backfillEmbeddings(pool, project, signal);
				})
				.catch((err) => {
					isIndexing = false;
					if (!signal.aborted) {
						console.error("[codebase-index] indexing error:", err);
					}
				})
				.finally(() => {
					backgroundTask = null;
					backgroundAbort = null;
				});
		}

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
		backgroundAbort?.abort();
		if (currentWatcher) {
			await currentWatcher.stop();
			currentWatcher = null;
		}
		if (backgroundTask !== null) {
			try {
				await Promise.race([
					backgroundTask,
					new Promise<void>((resolve) => setTimeout(resolve, 5000)),
				]);
			} catch {
				// Background task failed during shutdown
			}
		}
		await closePool();
		poolReady = false;
		await closeGraphPool();
		graphPoolReady = false;
	});

	// ── Auto-context injection (Cursor-style) ───────────────────────────────
	pi.on("before_agent_start", async (event, ctx) => {
		const pool = await ensurePool();
		if (!pool) return;

		const project = projectKey(ctx.cwd);
		const chunkCount = await getChunkCount(pool, project);
		if (chunkCount === 0) return;

		const prompt = event.prompt;
		if (!prompt || prompt.length < 10) return;

		// Skip auto-injection for casual questions that don't relate to code
		const codeWords = [
			"class",
			"function",
			"method",
			"variable",
			"import",
			"export",
			"interface",
			"refactor",
			"debug",
			"error",
			"fix",
			"test",
			"api",
			"database",
			"schema",
			"config",
			"file",
			"package",
			"build",
			"component",
			"handler",
			"service",
			"endpoint",
			"query",
			"search",
			"module",
			"type",
			"route",
		];
		const lower = prompt.toLowerCase();
		const hasCode = codeWords.some((w) => lower.includes(w));
		const firstWord = prompt.split(/\s+/)[0]?.toLowerCase();
		const isShortQuestion =
			!!firstWord && QUESTION_WORDS.has(firstWord) && prompt.split(/\s+/).length < 12;
		if (!hasCode && isShortQuestion) return;

		try {
			const results = await hybridSearch(pool, {
				query: prompt,
				mode: "auto",
				project,
				maxResults: 3,
				signal: AbortSignal.timeout(5000),
				rootDir: ctx.cwd,
			});

			if (results.length === 0) return;

			const MAX_SIG = 3;
			const lines = results
				.map((r) => {
					const sig = r.content.split("\n").slice(0, MAX_SIG).join("\n");
					const sym =
						r.symbol_type && r.symbol_name
							? ` ${r.symbol_type} ${r.symbol_name}`
							: "";
					return `  ${r.file_path}:${r.start_line}-${r.end_line}${sym}\n  ${sig}`;
				})
				.join("\n");
			const graphCtx = await buildGraphContext(results);

			return {
				message: {
					customType: "codebase-context",
					content: `[codebase-index] Relevant files for "${prompt.slice(0, 80)}":\n${lines}${graphCtx}`,
					display: false,
				},
			};
		} catch {
			// Silently fail — don't block the agent
		}
	});

	// ── Tool: codebase_search ────────────────────────────────────────────────
	pi.registerTool({
		name: "codebase_search",
		label: "Codebase Search",
		description:
			"Search the local codebase using regex, keyword, or semantic search. " +
			"Returns matching code chunks with file paths, line numbers, and symbol info.",
		promptSnippet:
			"Search local codebase by regex, keyword, or natural language",
		promptGuidelines: [
			"Use codebase_search for finding code in the LOCAL project workspace.",
			"Use code_search for searching the INTERNET for code examples and docs.",
			"For 'where is X defined?' or 'how is Y used?', prefer codebase_search with mode='auto'.",
		],
		parameters: Type.Object({
			query: Type.String({
				description: "Search query — regex, keyword, or natural language.",
			}),
			mode: Type.Optional(
				StringEnum(["auto", "regex", "keyword", "semantic"] as const, {
					description: "Search mode. Default: auto.",
					default: "auto",
				}),
			),
			language: Type.Optional(
				Type.String({ description: "Filter by language (e.g. 'typescript')." }),
			),
			maxResults: Type.Optional(
				Type.Number({
					description: "Max results. Default: 5.",
					default: 5,
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
							text: "codebase-index: PostgreSQL not available.",
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
							text: "codebase-index: No chunks indexed yet.",
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
					maxResults: params.maxResults ?? 5,
					signal,
					rootDir: ctx.cwd,
				});

				if (results.length === 0) {
					return {
						content: [
							{
								type: "text" as const,
								text: `No results for "${params.query}" (${chunkCount} chunks indexed).`,
							},
						],
						details: { query: params.query, mode: params.mode, chunkCount },
					};
				}

				const MAX_SIG = 5;
				const formatted = results
					.map((r) => {
						const sig = r.content.split("\n").slice(0, MAX_SIG).join("\n");
						const sym =
							r.symbol_type && r.symbol_name
								? ` ${r.symbol_type} ${r.symbol_name}`
								: "";
						return `--- ${r.file_path}:${r.start_line}-${r.end_line}${sym} ---\n${sig}`;
					})
					.join("\n\n");
				const graphCtx = await buildGraphContext(results);

				return {
					content: [
						{
							type: "text" as const,
							text: `${formatted}${graphCtx}`,
						},
					],
					details: {
						query: params.query,
						mode: params.mode || "auto",
						resultCount: results.length,
						chunkCount,
						results: results.map((r) => ({
							file_path: r.file_path,
							start_line: r.start_line,
							end_line: r.end_line,
							symbol_type: r.symbol_type,
							symbol_name: r.symbol_name,
							score: r.score,
						})),
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

	// ── Commands ────────────────────────────────────────────────────────────
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
			const hasKey = !!getDeepInfraKey();

			// Graph stats
			let graphInfo = "Graph: not available";
			try {
				const graphPool = await ensureGraphPool();
				if (graphPool) {
					const stats = await getGraphStats(graphPool);
					graphInfo = `Graph: ${stats.nodeCount} nodes, ${stats.edgeCount} edges`;
				}
			} catch {
				// Graph not available
			}

			ctx.ui.notify(
				[
					`Project: ${project}`,
					`Chunks: ${chunkCount} (${embeddedCount} with embeddings)`,
					`Semantic: ${hasKey ? "enabled" : "missing DEEPINFRA_API_KEY"}`,
					graphInfo,
				].join("\n"),
				"info",
			);
		},
	});
}
