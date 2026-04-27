/**
 * indexer.ts — File discovery + line-based chunking + content hashing
 *
 * Walks the workspace, reads source files, chunks them at function/class
 * boundaries using heuristic line scanning, and produces content-hashed
 * ChunkRow objects.
 */

import { createHash } from "node:crypto";
import { readdir, readFile, stat } from "node:fs/promises";
import { basename, extname, join, relative } from "node:path";
import { astChunkFile, astChunksToRows, hasAstSupport } from "./ast_chunker.js";
import type { ChunkRow } from "./db.js";
import { fileHasChanged } from "./db.js";

// ── Language detection ──────────────────────────────────────────────────────

const EXT_MAP: Record<string, string> = {
	".ts": "typescript",
	".tsx": "typescript",
	".js": "javascript",
	".jsx": "javascript",
	".mjs": "javascript",
	".cjs": "javascript",
	".py": "python",
	".rs": "rust",
	".go": "go",
	".java": "java",
	".kt": "kotlin",
	".kts": "kotlin",
	".swift": "swift",
	".c": "c",
	".h": "c",
	".cpp": "cpp",
	".cc": "cpp",
	".cxx": "cpp",
	".hpp": "cpp",
	".cs": "csharp",
	".rb": "ruby",
	".php": "php",
	".scala": "scala",
	".lua": "lua",
	".zig": "zig",
	".nim": "nim",
	".dart": "dart",
	".ex": "elixir",
	".exs": "elixir",
	".erl": "erlang",
	".hrl": "erlang",
	".hs": "haskell",
	".sql": "sql",
	".sh": "bash",
	".bash": "bash",
	".zsh": "bash",
	".toml": "toml",
	".yaml": "yaml",
	".yml": "yaml",
	".json": "json",
	".xml": "xml",
	".html": "html",
	".htm": "html",
	".css": "css",
	".scss": "scss",
	".less": "less",
	".vue": "vue",
	".svelte": "svelte",
};

const SKIP_DIRS = new Set([
	"node_modules",
	".git",
	".hg",
	".svn",
	"__pycache__",
	".tox",
	".mypy_cache",
	".pytest_cache",
	"venv",
	".venv",
	"env",
	".env",
	"dist",
	"build",
	"out",
	"target",
	".next",
	".nuxt",
	".cache",
	".turbo",
	"coverage",
	".nyc_output",
	"vendor",
	"Pods",
	".gradle",
	".idea",
	".vscode",
	".pi",
	"bin",
	"obj",
	"Debug",
	"Release",
	".dart_tool",
	".pub-cache",
]);

const SKIP_FILES = new Set([
	"package-lock.json",
	"yarn.lock",
	"pnpm-lock.yaml",
	"Gemfile.lock",
	"Cargo.lock",
	"go.sum",
	"poetry.lock",
	"pdm.lock",
	".DS_Store",
]);

const MAX_FILE_SIZE = 512 * 1024;
const MAX_CHUNK_LINES = 150;
const MIN_CHUNK_LINES = 6;
const OVERLAP_LINES = 10;

// ── Hashing ─────────────────────────────────────────────────────────────────

export function contentHash(content: string): string {
	return createHash("sha256").update(content).digest("hex").slice(0, 16);
}

// ── Language helpers ────────────────────────────────────────────────────────

export function detectLanguage(filePath: string): string | null {
	const name = basename(filePath);
	if (name === "Dockerfile" || name.startsWith("Dockerfile."))
		return "dockerfile";
	if (name === "Makefile" || name === "CMakeLists.txt") return "makefile";
	if (name === "Gemfile") return "ruby";
	if (name === "Cargo.toml") return "toml";
	return EXT_MAP[extname(filePath).toLowerCase()] ?? null;
}

export function isSourceFile(filePath: string): boolean {
	return detectLanguage(filePath) !== null;
}

// ── Symbol boundary detection ───────────────────────────────────────────────

interface SymbolMatch {
	lineIdx: number;
	name: string;
	type: string;
}

interface SymbolRule {
	pattern: RegExp;
	type: string;
}

// Table-driven symbol classification — avoids a long if-else chain
const SYMBOL_RULES: SymbolRule[] = [
	// Structural
	{
		pattern: /(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)/,
		type: "class",
	},
	{ pattern: /(?:export\s+)?interface\s+(\w+)/, type: "interface" },
	{ pattern: /(?:export\s+)?type\s+(\w+)\s*[<=]/, type: "type" },
	{ pattern: /(?:export\s+)?enum\s+(\w+)/, type: "enum" },
	{ pattern: /(?:export\s+)?namespace\s+(\w+)/, type: "module" },
	{ pattern: /(?:export\s+)?module\s+(\w+)/, type: "module" },
	// TS/JS functions
	{ pattern: /^(?:export\s+)?(?:async\s+)?function\s+(\w+)/, type: "function" },
	{
		pattern: /^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\S+\s*=>/,
		type: "function",
	},
	{
		pattern:
			/^(?:public|private|protected|static|abstract)\s+(?:async\s+)?(\w+)\s*\(/,
		type: "method",
	},
	// Python
	{ pattern: /^(?:async\s+)?def\s+(\w+)/, type: "function" },
	{ pattern: /^class\s+(\w+)/, type: "class" },
	// Rust
	{ pattern: /^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)/, type: "function" },
	{ pattern: /^(?:pub\s+)?struct\s+(\w+)/, type: "struct" },
	{ pattern: /^(?:pub\s+)?enum\s+(\w+)/, type: "enum" },
	{ pattern: /^(?:pub\s+)?trait\s+(\w+)/, type: "trait" },
	{ pattern: /^(?:pub\s+)?impl(?:<[^>]*>)?\s+(\w+)/, type: "impl" },
	{ pattern: /^(?:pub\s+)?mod\s+(\w+)/, type: "module" },
	// Go
	{ pattern: /^func\s+(?:\([^)]*\)\s+)?(\w+)/, type: "function" },
	{ pattern: /^type\s+(\w+)\s+(?:struct|interface)/, type: "class" },
	// Ruby
	{ pattern: /^def\s+(\w+)/, type: "function" },
	{ pattern: /^class\s+(\w+)\s*.*$/, type: "class" },
	{ pattern: /^module\s+(\w+)/, type: "module" },
	// Swift
	{
		pattern: /^(?:public|private|internal|open)?\s*func\s+(\w+)/,
		type: "function",
	},
	{
		pattern: /^(?:public|private|internal|open)?\s*class\s+(\w+)/,
		type: "class",
	},
	{
		pattern: /^(?:public|private|internal|open)?\s*struct\s+(\w+)/,
		type: "struct",
	},
	{
		pattern: /^(?:public|private|internal|open)?\s*enum\s+(\w+)/,
		type: "enum",
	},
	{
		pattern: /^(?:public|private|internal|open)?\s*protocol\s+(\w+)/,
		type: "interface",
	},
];

function classifySymbol(line: string): { name: string; type: string } | null {
	for (const rule of SYMBOL_RULES) {
		const match = rule.pattern.exec(line);
		if (match?.[1]) {
			return { name: match[1], type: rule.type };
		}
	}
	return null;
}

function findBoundaries(lines: string[]): SymbolMatch[] {
	const boundaries: SymbolMatch[] = [];
	for (let i = 0; i < lines.length; i++) {
		const sym = classifySymbol(lines[i]);
		if (sym) {
			boundaries.push({ lineIdx: i, name: sym.name, type: sym.type });
		}
	}
	return boundaries;
}

// ── Content-line filter ─────────────────────────────────────────────────────

const LINE_COMMENT_PREFIXES = ["//", "#", "--", ";", "%"];

function hasContent(line: string): boolean {
	const trimmed = line.trim();
	if (trimmed.length === 0) return false;
	return !LINE_COMMENT_PREFIXES.some((pfx) => trimmed.startsWith(pfx));
}

// ── Chunking ────────────────────────────────────────────────────────────────

interface ChunkParams {
	project: string;
	filePath: string;
	language: string;
	symbolType: string;
	symbolName: string | null;
	startLine: number;
	endLine: number;
	lines: string[];
}

function makeChunk(p: ChunkParams): ChunkRow {
	const content = p.lines.join("\n");
	return {
		project: p.project,
		file_path: p.filePath,
		language: p.language,
		symbol_type: p.symbolType,
		symbol_name: p.symbolName,
		start_line: p.startLine,
		end_line: p.endLine,
		content,
		content_hash: contentHash(content),
	};
}

function buildSymbolChunks(
	project: string,
	filePath: string,
	language: string,
	lines: string[],
	boundaries: SymbolMatch[],
): ChunkRow[] {
	const chunks: ChunkRow[] = [];

	// Skip preamble (comments + imports) — never useful for search

	// One chunk per symbol (sub-chunk large bodies)
	for (let i = 0; i < boundaries.length; i++) {
		const start = boundaries[i].lineIdx;
		const end =
			i + 1 < boundaries.length ? boundaries[i + 1].lineIdx : lines.length;
		const bodyLen = end - start;

		if (bodyLen > MAX_CHUNK_LINES) {
			for (let s = start; s < end; s += MAX_CHUNK_LINES - OVERLAP_LINES) {
				const subEnd = Math.min(s + MAX_CHUNK_LINES, end);
				chunks.push(
					makeChunk({
						project,
						filePath,
						language,
						symbolType: boundaries[i].type,
						symbolName: boundaries[i].name,
						startLine: s + 1,
						endLine: subEnd,
						lines: lines.slice(s, subEnd),
					}),
				);
			}
		} else {
			const symbolLines = lines.slice(start, end);
			if (symbolLines.length > 0) {
				chunks.push(
					makeChunk({
						project,
						filePath,
						language,
						symbolType: boundaries[i].type,
						symbolName: boundaries[i].name,
						startLine: start + 1,
						endLine: end,
						lines: symbolLines,
					}),
				);
			}
		}
	}

	return chunks;
}

function buildFallbackChunks(
	project: string,
	filePath: string,
	language: string,
	lines: string[],
): ChunkRow[] {
	const chunks: ChunkRow[] = [];
	for (
		let start = 0;
		start < lines.length;
		start += MAX_CHUNK_LINES - OVERLAP_LINES
	) {
		const end = Math.min(start + MAX_CHUNK_LINES, lines.length);
		const subLines = lines.slice(start, end);
		if (start > 0 && subLines.filter(hasContent).length < MIN_CHUNK_LINES)
			continue;
		chunks.push(
			makeChunk({
				project,
				filePath,
				language,
				symbolType: "block",
				symbolName: null,
				startLine: start + 1,
				endLine: end,
				lines: subLines,
			}),
		);
	}
	return chunks;
}

async function chunkFile(
	project: string,
	filePath: string,
	language: string,
	content: string,
): Promise<ChunkRow[]> {
	// Try AST chunking first (tree-sitter)
	if (hasAstSupport(filePath)) {
		try {
			const astChunks = await astChunkFile(filePath, language, content);
			if (astChunks && astChunks.length > 0) {
				return astChunksToRows(project, filePath, language, astChunks);
			}
		} catch {
			// Fall through to regex chunking
		}
	}

	// Fallback: regex-based heuristic chunking
	const lines = content.split("\n");
	const boundaries = findBoundaries(lines);
	if (boundaries.length === 0) {
		return buildFallbackChunks(project, filePath, language, lines);
	}
	return buildSymbolChunks(project, filePath, language, lines, boundaries);
}

// ── File collection ─────────────────────────────────────────────────────────

async function collectSourceFiles(
	rootDir: string,
	signal?: AbortSignal,
): Promise<string[]> {
	const files: string[] = [];

	async function walk(dir: string): Promise<void> {
		if (signal?.aborted) return;
		let entries: import("node:fs").Dirent[];
		try {
			entries = await readdir(dir, { withFileTypes: true });
		} catch {
			return;
		}

		for (const entry of entries) {
			if (signal?.aborted) return;
			const fullPath = join(dir, entry.name);
			if (entry.isDirectory()) {
				if (shouldSkipEntry(entry.name)) continue;
				await walk(fullPath);
			} else if (entry.isFile()) {
				if (!shouldIndexFile(entry.name, fullPath)) continue;
				files.push(fullPath);
			}
		}
	}

	await walk(rootDir);
	return files;
}

function shouldSkipEntry(name: string): boolean {
	if (SKIP_DIRS.has(name)) return true;
	if (name.startsWith(".") && name !== ".github") return true;
	return false;
}

function shouldIndexFile(name: string, fullPath: string): boolean {
	if (SKIP_FILES.has(name)) return false;
	if (name.startsWith(".") && !name.endsWith(".ts")) return false;
	if (!isSourceFile(fullPath)) return false;
	return true;
}

// ── Single file processing ──────────────────────────────────────────────────

async function processFile(
	filePath: string,
	project: string,
	rootDir: string,
	pool?: import("pg").Pool,
): Promise<ChunkRow[]> {
	const info = await stat(filePath);
	if (info.size > MAX_FILE_SIZE) return [];

	const content = await readFile(filePath, "utf-8");
	if (content.trim().length === 0) return [];

	const language = detectLanguage(filePath);
	if (!language) return [];

	const relPath = relative(rootDir, filePath);

	// Merkle-tree check: skip if file content hasn't changed
	if (pool) {
		const hash = contentHash(content);
		const changed = await fileHasChanged(pool, project, relPath, hash);
		if (!changed) return [];
	}

	return chunkFile(project, relPath, language, content);
}

// ── Public API ──────────────────────────────────────────────────────────────

export interface IndexingOptions {
	project: string;
	rootDir: string;
	signal?: AbortSignal;
	onProgress?: (file: string, chunks: number) => void;
	pool?: import("pg").Pool;
}

export interface IndexingResult {
	totalFiles: number;
	totalChunks: number;
	skippedFiles: number;
	errors: string[];
	chunks: ChunkRow[];
}

/** Index all source files in the workspace and return collected chunks. */
export async function indexWorkspace(
	options: IndexingOptions,
): Promise<IndexingResult> {
	const { project, rootDir, signal, onProgress } = options;
	const result: IndexingResult = {
		totalFiles: 0,
		totalChunks: 0,
		skippedFiles: 0,
		errors: [],
		chunks: [],
	};

	const files = await collectSourceFiles(rootDir, signal);
	result.totalFiles = files.length;

	const BATCH_SIZE = 20;
	for (let i = 0; i < files.length; i += BATCH_SIZE) {
		if (signal?.aborted) break;

		const batch = files.slice(i, i + BATCH_SIZE);
		const batchResults = await Promise.allSettled(
			batch.map(async (filePath) => {
				const chunks = await processFile(filePath, project, rootDir, options.pool);
				if (chunks.length > 0) {
					const relPath = relative(rootDir, filePath);
					onProgress?.(relPath, chunks.length);
				}
				return chunks;
			}),
		);

		for (const settled of batchResults) {
			if (settled.status === "fulfilled") {
				result.chunks.push(...settled.value);
				result.totalChunks += settled.value.length;
				if (settled.value.length === 0) result.skippedFiles++;
			} else {
				result.errors.push(String(settled.reason));
			}
		}
	}

	return result;
}

/** Index a single file and return its chunks. */
export async function indexFile(
	project: string,
	rootDir: string,
	filePath: string,
	pool?: import("pg").Pool,
): Promise<ChunkRow[]> {
	return processFile(filePath, project, rootDir, pool);
}
