/**
 * ast_chunker.ts — Tree-sitter based code chunking
 *
 * Parses source files with tree-sitter ASTs to extract chunks at real
 * symbol boundaries (functions, classes, methods, interfaces, etc).
 * Falls back to line-based chunking for unsupported languages.
 */

import { createHash } from "node:crypto";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

import type { ChunkRow } from "./db.js";

// ── Types ───────────────────────────────────────────────────────────────────

interface TsNode {
	type: string;
	text: string;
	startPosition: { row: number; column: number };
	endPosition: { row: number; column: number };
	namedChildren: TsNode[];
	isNamed: boolean;
	childForFieldName(name: string): TsNode | null;
	delete(): void;
}

interface AstChunk {
	symbolType: string;
	symbolName: string | null;
	startLine: number;
	endLine: number;
	content: string;
}

// ── Grammar registry ────────────────────────────────────────────────────────

interface GrammarEntry {
	wasmPath: string;
	chunkTypes: string[];
	wrapperTypes: string[];
	methodTypes: string[];
}

const GRAMMARS: Record<string, GrammarEntry> = {
	typescript: {
		wasmPath: "tree-sitter-typescript/tree-sitter-typescript.wasm",
		chunkTypes: [
			"function_declaration",
			"class_declaration",
			"interface_declaration",
			"type_alias_declaration",
			"enum_declaration",
			"lexical_declaration",
			"variable_declaration",
			"abstract_class_declaration",
		],
		wrapperTypes: ["export_statement", "ambient_declaration"],
		methodTypes: ["method_definition", "arrow_function"],
	},
	tsx: {
		wasmPath: "tree-sitter-typescript/tree-sitter-tsx.wasm",
		chunkTypes: [
			"function_declaration",
			"class_declaration",
			"interface_declaration",
			"type_alias_declaration",
			"enum_declaration",
			"lexical_declaration",
			"variable_declaration",
			"abstract_class_declaration",
		],
		wrapperTypes: ["export_statement", "ambient_declaration"],
		methodTypes: ["method_definition", "arrow_function"],
	},
	javascript: {
		wasmPath: "tree-sitter-javascript/tree-sitter-javascript.wasm",
		chunkTypes: [
			"function_declaration",
			"class_declaration",
			"lexical_declaration",
			"variable_declaration",
		],
		wrapperTypes: ["export_statement"],
		methodTypes: ["method_definition", "arrow_function"],
	},
	python: {
		wasmPath: "tree-sitter-python/tree-sitter-python.wasm",
		chunkTypes: [
			"function_definition",
			"class_definition",
			"decorated_definition",
		],
		wrapperTypes: [],
		methodTypes: ["function_definition"],
	},
	rust: {
		wasmPath: "tree-sitter-rust/tree-sitter-rust.wasm",
		chunkTypes: [
			"function_item",
			"struct_item",
			"enum_item",
			"impl_item",
			"trait_item",
			"type_item",
			"mod_item",
			"const_item",
			"static_item",
		],
		wrapperTypes: [],
		methodTypes: ["function_item"],
	},
	go: {
		wasmPath: "tree-sitter-go/tree-sitter-go.wasm",
		chunkTypes: [
			"function_declaration",
			"method_declaration",
			"type_declaration",
		],
		wrapperTypes: [],
		methodTypes: [],
	},
	java: {
		wasmPath: "tree-sitter-java/tree-sitter-java.wasm",
		chunkTypes: [
			"class_declaration",
			"interface_declaration",
			"enum_declaration",
			"record_declaration",
			"method_declaration",
		],
		wrapperTypes: [],
		methodTypes: ["method_declaration"],
	},
	ruby: {
		wasmPath: "tree-sitter-ruby/tree-sitter-ruby.wasm",
		chunkTypes: ["method", "singleton_method", "class", "module"],
		wrapperTypes: [],
		methodTypes: ["method", "singleton_method"],
	},
};

const EXT_TO_GRAMMAR: Record<string, string> = {
	".ts": "typescript",
	".tsx": "tsx",
	".js": "javascript",
	".jsx": "javascript",
	".mjs": "javascript",
	".cjs": "javascript",
	".py": "python",
	".rs": "rust",
	".go": "go",
	".java": "java",
	".rb": "ruby",
};

// ── Parser singleton ────────────────────────────────────────────────────────

// We store the dynamic imports lazily — types are loose because web-tree-sitter
// uses a WASM-based API that doesn't map cleanly to static types
let ParserConstructor:
	| (new () => {
			setLanguage(lang: unknown): void;
			parse(source: string): { rootNode: TsNode; delete(): void } | null;
	  })
	| null = null;
let LanguageLoader: { load(path: string): Promise<unknown> } | null = null;
const languageCache = new Map<string, unknown>();

async function ensureParser(): Promise<boolean> {
	if (ParserConstructor) return true;
	try {
		// web-tree-sitter exports Parser and Language at top level (CJS)
		const wt = require("web-tree-sitter") as {
			Parser: new () => {
				setLanguage(lang: unknown): void;
				parse(source: string): { rootNode: TsNode; delete(): void } | null;
			};
			Language: { load(path: string): Promise<unknown> };
		};
		await wt.Parser.init();
		ParserConstructor = wt.Parser;
		LanguageLoader = wt.Language;
		return true;
	} catch (err) {
		console.error(
			"[ast_chunker] web-tree-sitter init failed:",
			(err as Error).message,
		);
		return false;
	}
}

async function loadLanguage(grammarName: string): Promise<Record<string, unknown> | null> {
	if (languageCache.has(grammarName))
		return languageCache.get(grammarName) ?? null;

	const grammar = GRAMMARS[grammarName];
	if (!grammar || !LanguageLoader) return null;

	const wasmPath = join(__dirname, "node_modules", grammar.wasmPath);

	try {
		const lang = await LanguageLoader.load(wasmPath);
		languageCache.set(grammarName, lang);
		return lang;
	} catch (err) {
		console.error(
			"[ast_chunker] failed to load %s:",
			grammarName,
			(err as Error).message,
		);
		return null;
	}
}

// ── AST walking ─────────────────────────────────────────────────────────────

function getSymbolName(node: TsNode): string | null {
	const nameNode =
		node.childForFieldName("name") ?? node.childForFieldName("value");
	return nameNode?.text ?? null;
}

const NODE_TYPE_MAP: Record<string, string> = {
	function_declaration: "function",
	function_definition: "function",
	function_item: "function",
	method_definition: "method",
	method_declaration: "method",
	method: "method",
	singleton_method: "method",
	class_declaration: "class",
	class_definition: "class",
	class: "class",
	abstract_class_declaration: "class",
	struct_item: "struct",
	interface_declaration: "interface",
	impl_item: "impl",
	trait_item: "trait",
	type_alias_declaration: "type",
	type_declaration: "type",
	type_item: "type",
	enum_declaration: "enum",
	enum_item: "enum",
	module: "module",
	mod_item: "module",
	lexical_declaration: "variable",
	variable_declaration: "variable",
	const_item: "constant",
	static_item: "constant",
	record_declaration: "class",
	decorated_definition: "function",
};

function mapSymbolType(nodeType: string): string {
	const mapped = NODE_TYPE_MAP[nodeType];
	if (mapped) return mapped;
	return "unknown";
}

function unwrapNode(node: TsNode, wrapperTypes: string[]): TsNode {
	if (!wrapperTypes.includes(node.type)) return node;
	for (const child of node.namedChildren) {
		if (child.isNamed) return child;
	}
	return node;
}

// Preamble extraction disabled — file headers/imports are not useful for search
function extractPreamble(): AstChunk | null {
	return null;
}

function extractMethods(
	classNode: TsNode,
	lines: string[],
	methodTypes: string[],
): AstChunk[] {
	const chunks: AstChunk[] = [];
	const body = classNode.childForFieldName("body");
	if (!body) return chunks;

	for (const member of body.namedChildren) {
		if (methodTypes.includes(member.type)) {
			const mStart = member.startPosition.row + 1;
			const mEnd = member.endPosition.row + 1;
			chunks.push({
				symbolType: "method",
				symbolName: getSymbolName(member),
				startLine: mStart,
				endLine: mEnd,
				content: lines.slice(mStart - 1, mEnd).join("\n"),
			});
		}
	}
	return chunks;
}

function extractSymbolChunks(
	rootNode: TsNode,
	lines: string[],
	grammar: GrammarEntry,
): AstChunk[] {
	const chunks: AstChunk[] = [];
	let lastEndLine = 0;

	for (const child of rootNode.namedChildren) {
		const actualNode = unwrapNode(child, grammar.wrapperTypes);

		if (!grammar.chunkTypes.includes(actualNode.type)) continue;

		const startLine = actualNode.startPosition.row + 1;
		const endLine = actualNode.endPosition.row + 1;
		const symbolType = mapSymbolType(actualNode.type);

		let symbolName = getSymbolName(actualNode);
		if (
			(actualNode.type === "lexical_declaration" ||
				actualNode.type === "variable_declaration") &&
			!symbolName
		) {
			const declarator = actualNode.childForFieldName("declarator");
			if (declarator) {
				const nameNode = declarator.childForFieldName("name") ?? declarator;
				symbolName = nameNode.text.split(/\s/)[0] ?? null;
			}
		}

		const preamble = extractPreamble(lines, startLine, lastEndLine);
		if (preamble) chunks.push(preamble);

		const content = lines.slice(startLine - 1, endLine).join("\n");
		chunks.push({ symbolType, symbolName, startLine, endLine, content });

		lastEndLine = endLine;

		if (
			symbolType === "class" ||
			symbolType === "struct" ||
			symbolType === "impl"
		) {
			chunks.push(...extractMethods(actualNode, lines, grammar.methodTypes));
		}
	}

	return chunks;
}

// ── Public API ──────────────────────────────────────────────────────────────

export function hasAstSupport(filePath: string): boolean {
	const ext = getExt(filePath);
	const grammarName = EXT_TO_GRAMMAR[ext];
	return grammarName !== undefined && GRAMMARS[grammarName] !== undefined;
}

function getExt(filePath: string): string {
	const parts = filePath.split(".");
	const last = parts.at(-1);
	const secondLast = parts.at(-2);
	if (parts.length >= 3 && secondLast && secondLast.length <= 3) {
		return `.${secondLast}.${last}`;
	}
	return last ? `.${last}` : "";
}

export async function astChunkFile(
	filePath: string,
	language: string,
	sourceCode: string,
): Promise<AstChunk[] | null> {
	const ext = getExt(filePath);
	const grammarName = EXT_TO_GRAMMAR[ext] ?? language;
	const grammar = GRAMMARS[grammarName];
	if (!grammar) return null;

	const ready = await ensureParser();
	if (!ready) return null;

	const lang = await loadLanguage(grammarName);
	if (!lang) return null;

	try {
		// ParserConstructor is guaranteed non-null after ensureParser() check above
		const Ctor = ParserConstructor;
		if (!Ctor) return null;
		const parser = new Ctor();
		parser.setLanguage(lang);
		const tree = parser.parse(sourceCode);
		if (!tree) return null;

		const lines = sourceCode.split("\n");
		const chunks = extractSymbolChunks(tree.rootNode, lines, grammar);
		tree.delete();
		return chunks.length > 0 ? chunks : null;
	} catch (err) {
		console.error(
			"[ast_chunker] parse error %s:",
			filePath,
			(err as Error).message,
		);
		return null;
	}
}

export function astChunksToRows(
	project: string,
	filePath: string,
	language: string,
	chunks: AstChunk[],
): ChunkRow[] {
	const MIN_CONTENT_LINES = 6;
	return chunks
		.filter((chunk) => chunk.content.split("\n").length >= MIN_CONTENT_LINES)
		.map((chunk) => ({
			project,
			file_path: filePath,
			language,
			symbol_type: chunk.symbolType,
			symbol_name: chunk.symbolName,
			start_line: chunk.startLine,
			end_line: chunk.endLine,
			content: chunk.content,
			content_hash: createHash("sha256")
				.update(chunk.content)
				.digest("hex")
				.slice(0, 16),
		}));
}
