/**
 * watcher.ts — File watcher for incremental re-indexing
 *
 * Uses chokidar to watch for file changes and triggers incremental
 * re-indexing of changed files only.
 */

import { relative, resolve } from "node:path";
import type { Pool } from "pg";
import {
	deleteFileChunks,
	getChunksWithoutEmbeddings,
	updateEmbeddings,
	upsertChunks,
} from "./db.js";
import { generateEmbeddings } from "./embed.js";
import { indexFile, isSourceFile } from "./indexer.js";

// Try to import chokidar — it's an optional dependency
let chokidar: typeof import("chokidar") | null = null;
try {
	chokidar = await import("chokidar");
} catch {
	// chokidar not available — watcher will be disabled
}

export interface WatcherOptions {
	pool: Pool;
	project: string;
	rootDir: string;
	/** Debounce interval in ms (default 1000) */
	debounceMs?: number;
	onReindex?: (file: string, chunks: number) => void;
}

interface PendingChange {
	type: "change" | "unlink";
	filePath: string;
}

export interface FileWatcher {
	stop(): Promise<void>;
}

export async function startWatcher(
	options: WatcherOptions,
): Promise<FileWatcher | null> {
	if (!chokidar) {
		console.log(
			"[codebase-index] chokidar not available, file watcher disabled",
		);
		return null;
	}

	const { pool, project, rootDir, debounceMs = 1000, onReindex } = options;
	let pending: PendingChange[] = [];
	let debounceTimer: ReturnType<typeof setTimeout> | null = null;

	async function backfillEmbeddings(): Promise<void> {
		try {
			const unembedded = await getChunksWithoutEmbeddings(pool, project, 64);
			if (unembedded.length === 0) return;
			const texts = unembedded.map((c) => c.content);
			const { embeddings } = await generateEmbeddings(texts);
			const map = new Map<number, number[]>();
			for (let i = 0; i < unembedded.length; i++) {
				const id = unembedded[i].id;
				if (id !== undefined && embeddings[i]) map.set(id, embeddings[i]);
			}
			await updateEmbeddings(pool, map);
		} catch (embedErr) {
			console.error(
				"[codebase-index] watcher embedding backfill failed:",
				(embedErr as Error).message,
			);
		}
	}

	async function processFileChange(change: PendingChange): Promise<void> {
		if (change.type === "unlink") {
			await deleteFileChunks(pool, project, relative(rootDir, change.filePath));
			return;
		}
		const chunks = await indexFile(project, rootDir, change.filePath, pool);
		if (chunks.length === 0) return;
		await upsertChunks(pool, chunks);
		onReindex?.(relative(rootDir, change.filePath), chunks.length);
		await backfillEmbeddings();
	}

	async function processPending(): Promise<void> {
		const changes = pending.slice();
		pending = [];
		debounceTimer = null;
		if (changes.length === 0) return;

		// Deduplicate — keep last operation per file
		const byFile = new Map<string, PendingChange>();
		for (const change of changes) byFile.set(change.filePath, change);

		for (const change of byFile.values()) {
			try {
				await processFileChange(change);
			} catch (err) {
				console.error(
					"[codebase-index] error processing %s:",
					change.filePath,
					(err as Error).message,
				);
			}
		}
	}

	function scheduleFlush(): void {
		if (debounceTimer) clearTimeout(debounceTimer);
		debounceTimer = setTimeout(processPending, debounceMs);
	}

	// Watch source files only
	const watcher = chokidar.watch("**/*", {
		cwd: rootDir,
		ignored: (path: string) => {
			if (path.includes("node_modules")) return true;
			if (path.includes(".git/")) return true;
			if (path.includes("dist/")) return true;
			if (path.includes("build/")) return true;
			if (path.includes("target/")) return true;
			if (path.includes(".pi/")) return true;
			const fullPath = resolve(rootDir, path);
			return !isSourceFile(fullPath);
		},
		persistent: true,
		ignoreInitial: true,
		awaitWriteFinish: {
			stabilityThreshold: 500,
			pollInterval: 100,
		},
	});

	watcher.on("change", (path: string) => {
		pending.push({ type: "change", filePath: resolve(rootDir, path) });
		scheduleFlush();
	});

	watcher.on("unlink", (path: string) => {
		pending.push({ type: "unlink", filePath: resolve(rootDir, path) });
		scheduleFlush();
	});

	watcher.on("error", (err: unknown) => {
		const msg = err instanceof Error ? err.message : JSON.stringify(err);
		console.error("[codebase-index] watcher error:", msg);
	});

	return {
		async stop() {
			if (debounceTimer) clearTimeout(debounceTimer);
			await watcher.close();
			await processPending();
		},
	};
}
