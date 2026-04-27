/**
 * watcher.ts — File watcher for incremental re-indexing
 *
 * Uses chokidar to watch for file changes and triggers incremental
 * re-indexing of changed files only.
 */

import { relative, resolve } from "node:path";
import type { Pool } from "pg";
import { deleteFileChunks, upsertChunks } from "./db.js";
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

	async function processPending(): Promise<void> {
		const changes = pending.slice();
		pending = [];
		debounceTimer = null;

		if (changes.length === 0) return;

		// Deduplicate — keep last operation per file
		const byFile = new Map<string, PendingChange>();
		for (const change of changes) {
			byFile.set(change.filePath, change);
		}

		for (const change of byFile.values()) {
			try {
				if (change.type === "unlink") {
					const relPath = relative(rootDir, change.filePath);
					await deleteFileChunks(pool, project, relPath);
				} else {
					const chunks = await indexFile(project, rootDir, change.filePath);
					if (chunks.length > 0) {
						await upsertChunks(pool, chunks);
						const relPath = relative(rootDir, change.filePath);
						onReindex?.(relPath, chunks.length);
					}
				}
			} catch (err) {
				console.error(
					`[codebase-index] error processing ${change.filePath}:`,
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
