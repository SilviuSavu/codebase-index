/**
 * edge_extractor.ts — Extract graph edges from source code
 *
 * Walks source files to extract structural relationships:
 * - CONTAINS: file → module → class → method
 * - IMPORTS: file → imported module/symbol
 * - CALLS: function → called function
 * - EXTENDS: class → parent class
 * - IMPLEMENTS: class → interface
 */

import type { EdgeSpec, GraphNode, NodeType } from "./graph.js";

// ── Types ───────────────────────────────────────────────────────────────────

export interface ExtractedGraph {
	nodes: GraphNode[];
	edges: EdgeSpec[];
}

interface ImportInfo {
	sourcePath: string;
	symbolName: string;
	alias?: string;
	isDefault?: boolean;
}

interface CallInfo {
	functionName: string;
	receiver?: string;
}

interface HeritageInfo {
	name: string;
	type: "extends" | "implements";
}

interface HeritageClause {
	className: string;
	heritage: HeritageInfo[];
}

interface SymbolDef {
	name: string;
	type: NodeType;
	startLine: number;
	parentSymbol?: string;
	visibility?: string;
	isAsync?: boolean;
	isStatic?: boolean;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function collectMatches(re: RegExp, text: string): RegExpExecArray[] {
	const results: RegExpExecArray[] = [];
	let m: RegExpExecArray | null = re.exec(text);
	while (m !== null) { results.push(m); m = re.exec(text); }
	return results;
}

function resolveImportPath(specifier: string, fromFilePath: string): string {
	const spec = specifier.replaceAll(/^['"]|['"]$/g, "");
	if (!spec.startsWith("./") && !spec.startsWith("../")) return spec;
	const parts = fromFilePath.split("/");
	parts.pop();
	const resolved = [...parts, ...spec.split("/")];
	const normalized: string[] = [];
	for (const part of resolved) {
		if (part === "..") normalized.pop();
		else if (part !== "." && part !== "") normalized.push(part);
	}
	return normalized.join("/");
}

function fileName(filePath: string): string {
	return filePath.split("/").pop() ?? filePath;
}

// ── Import extraction ───────────────────────────────────────────────────────

function extractTsImports(content: string, filePath: string): ImportInfo[] {
	const imports: ImportInfo[] = [];
	for (const m of collectMatches(/import\s+(?:(?:\{[^}]*\}|\*\s+as\s+\w+|\w+)\s+from\s+)?['"]([^'"]+)['"]/g, content)) {
		const source = m[1];
		const snippet = content.slice(m.index, m.index + 200);
		const named = snippet.match(/\{([^}]+)\}/);
		if (named) {
			for (const sym of named[1].split(",")) {
				const parts = sym.trim().split(/\s+as\s+/);
				imports.push({ sourcePath: resolveImportPath(source, filePath), symbolName: parts[0].trim(), alias: parts[1]?.trim() });
			}
		} else {
			const def = snippet.match(/import\s+(\w+)/);
			imports.push({ sourcePath: resolveImportPath(source, filePath), symbolName: def?.[1] ?? "*", isDefault: !!def });
		}
	}
	for (const m of collectMatches(/(?:const|let|var)\s+(\w+)\s*=\s*require\s*\(\s*['"]([^'"]+)['"]\s*\)/g, content)) {
		imports.push({ sourcePath: resolveImportPath(m[2], filePath), symbolName: m[1] });
	}
	return imports;
}

function extractPythonImports(content: string): ImportInfo[] {
	const imports: ImportInfo[] = [];
	for (const m of collectMatches(/from\s+([\w.]+)\s+import\s+(.+)/g, content)) {
		for (const sym of m[2].split(",")) {
			const parts = sym.trim().split(/\s+as\s+/);
			imports.push({ sourcePath: m[1].replaceAll(".", "/"), symbolName: parts[0].trim(), alias: parts[1]?.trim() });
		}
	}
	for (const m of collectMatches(/import\s+([\w.]+)(?:\s+as\s+(\w+))?/g, content)) {
		imports.push({ sourcePath: m[1].replaceAll(".", "/"), symbolName: m[1].split(".").at(-1) ?? m[1], alias: m[2] ?? undefined });
	}
	return imports;
}

function extractRustImports(content: string): ImportInfo[] {
	const imports: ImportInfo[] = [];
	for (const m of collectMatches(/use\s+([\w:]+)(?:::\{([^}]+)\})?;/g, content)) {
		const path = m[1].replaceAll("::", "/");
		if (m[2]) {
			for (const sym of m[2].split(",")) imports.push({ sourcePath: path, symbolName: sym.trim() });
		} else {
			imports.push({ sourcePath: path.replace(/\/[^/]+$/, ""), symbolName: path.split("/").pop() ?? path });
		}
	}
	return imports;
}

function extractGoImports(content: string): ImportInfo[] {
	const imports: ImportInfo[] = [];
	for (const m of collectMatches(/import\s+(?:\(\s*([\s\S]*?)\s*\)|"([^"]+)")/g, content)) {
		if (m[2]) {
			imports.push({ sourcePath: m[2], symbolName: "*" });
		} else if (m[1]) {
			for (const line of m[1].split("\n")) {
				const lm = line.match(/"([^"]+)"/);
				if (lm) imports.push({ sourcePath: lm[1], symbolName: "*" });
			}
		}
	}
	return imports;
}

function extractRubyImports(content: string, filePath: string): ImportInfo[] {
	const imports: ImportInfo[] = [];
	for (const m of collectMatches(/require(?:_relative)?\s+['"]([^'"]+)['"]/g, content)) {
		imports.push({ sourcePath: resolveImportPath(m[1], filePath), symbolName: "*" });
	}
	return imports;
}

function extractJavaImports(content: string): ImportInfo[] {
	const imports: ImportInfo[] = [];
	for (const m of collectMatches(/import\s+(?:static\s+)?([\w.]+)(?:\.\*)?;/g, content)) {
		const parts = m[1].split(".");
		imports.push({ sourcePath: parts.slice(0, -1).join("/"), symbolName: parts.at(-1) ?? m[1] });
	}
	return imports;
}

function extractImports(content: string, filePath: string, language: string): ImportInfo[] {
	switch (language) {
		case "typescript": case "tsx": case "javascript": case "vue": case "svelte":
			return extractTsImports(content, filePath);
		case "python": return extractPythonImports(content);
		case "rust": return extractRustImports(content);
		case "go": return extractGoImports(content);
		case "ruby": return extractRubyImports(content, filePath);
		case "java": return extractJavaImports(content);
		default: return [];
	}
}

// ── Call extraction ─────────────────────────────────────────────────────────

const CALL_KEYWORDS = new Set([
	"if", "else", "for", "while", "switch", "case", "catch", "return",
	"function", "class", "interface", "type", "enum", "const", "let",
	"var", "new", "throw", "typeof", "instanceof", "import", "export",
	"async", "await", "yield", "def", "fn", "pub", "self", "super",
	"this", "constructor", "extends", "implements",
]);

function extractCalls(content: string): CallInfo[] {
	const calls: CallInfo[] = [];
	const seen = new Set<string>();
	for (const m of collectMatches(/(?:(\w+)\s*\.\s*)?(\w+)\s*\(/g, content)) {
		const fn = m[2];
		if (CALL_KEYWORDS.has(fn)) continue;
		const key = `${m[1] ?? ""}.${fn}`;
		if (seen.has(key)) continue;
		seen.add(key);
		calls.push({ functionName: fn, receiver: m[1] || undefined });
	}
	return calls;
}

// ── Heritage extraction ──────────���──────────────────────────────────────���───

function extractTsHeritage(content: string): HeritageClause[] {
	const results: HeritageClause[] = [];
	for (const m of collectMatches(/(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?/g, content)) {
		const heritage: HeritageInfo[] = [];
		if (m[2]) heritage.push({ name: m[2], type: "extends" });
		if (m[3]) {
			for (const iface of m[3].split(",")) {
				const t = iface.trim();
				if (t) heritage.push({ name: t, type: "implements" });
			}
		}
		if (heritage.length > 0) results.push({ className: m[1], heritage });
	}
	return results;
}

function extractPythonHeritage(content: string): HeritageClause[] {
	const results: HeritageClause[] = [];
	for (const m of collectMatches(/class\s+(\w+)\s*\(([^)]+)\)/g, content)) {
		const heritage: HeritageInfo[] = [];
		for (const base of m[2].split(",")) {
			const t = base.trim();
			if (t && t !== "object" && t !== "ABC") heritage.push({ name: t, type: "extends" });
		}
		if (heritage.length > 0) results.push({ className: m[1], heritage });
	}
	return results;
}

function extractRustHeritage(content: string): HeritageClause[] {
	const results: HeritageClause[] = [];
	for (const m of collectMatches(/impl\s+(?:<[^>]*>\s*)?(\w+)\s+for\s+(\w+)/g, content)) {
		results.push({ className: m[2], heritage: [{ name: m[1], type: "implements" }] });
	}
	return results;
}

function extractGoHeritage(content: string): HeritageClause[] {
	const results: HeritageClause[] = [];
	for (const m of collectMatches(/type\s+(\w+)\s+interface\s*\{([^}]*)\}/g, content)) {
		const heritage: HeritageInfo[] = [];
		for (const line of m[2].split("\n")) {
			const t = line.trim();
			if (t && /^[A-Z]\w*$/.test(t)) heritage.push({ name: t, type: "extends" });
		}
		if (heritage.length > 0) results.push({ className: m[1], heritage });
	}
	return results;
}

function extractHeritage(content: string, language: string): HeritageClause[] {
	switch (language) {
		case "typescript": case "tsx": case "javascript": return extractTsHeritage(content);
		case "python": return extractPythonHeritage(content);
		case "rust": return extractRustHeritage(content);
		case "go": return extractGoHeritage(content);
		default: return [];
	}
}

// ── Symbol extraction ───────────────────────────────────────────────────────

function extractTsSymbols(content: string): SymbolDef[] {
	return extractSymbolsGeneric(content, {
		classRe: /(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)/,
		ifaceRe: /(?:export\s+)?interface\s+(\w+)/,
		typeRe: /(?:export\s+)?type\s+(\w+)\s*[<=]/,
		funcRe: /^(?:export\s+)?(?:async\s+)?function\s+(\w+)/,
		arrowRe: /^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?/,
		methodRe: /(?:(public|private|protected)\s+)?(?:(static)\s+)?(?:async\s+)?(\w+)\s*\(/,
	});
}

function extractPythonSymbols(content: string): SymbolDef[] {
	const symbols: SymbolDef[] = [];
	let currentClass: string | undefined;
	let classIndent = 0;

	for (const [idx, line] of content.split("\n").entries()) {
		const ln = idx + 1;
		const indent = line.search(/\S/);
		if (currentClass && indent <= classIndent) currentClass = undefined;

		const cm = line.match(/^(\s*)class\s+(\w+)/);
		if (cm) { currentClass = cm[2]; classIndent = cm[1].length; symbols.push({ name: cm[2], type: "Class", startLine: ln }); continue; }

		const fm = line.match(/^(\s*)(?:async\s+)?def\s+(\w+)/);
		if (fm) {
			symbols.push({ name: fm[2], type: currentClass ? "Method" : "Function", startLine: ln, parentSymbol: currentClass, visibility: fm[2].startsWith("_") ? "private" : "public", isAsync: line.includes("async") });
		}
	}
	return symbols;
}

function extractRustSymbols(content: string): SymbolDef[] {
	const symbols: SymbolDef[] = [];
	for (const [idx, line] of content.split("\n").entries()) {
		const ln = idx + 1;
		const fm = line.match(/^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)/);
		if (fm) { symbols.push({ name: fm[1], type: "Function", startLine: ln, visibility: line.includes("pub") ? "public" : "private", isAsync: line.includes("async") }); continue; }
		const sm = line.match(/^(?:pub\s+)?struct\s+(\w+)/);
		if (sm) { symbols.push({ name: sm[1], type: "Struct", startLine: ln, visibility: line.includes("pub") ? "public" : "private" }); continue; }
		const tm = line.match(/^(?:pub\s+)?trait\s+(\w+)/);
		if (tm) { symbols.push({ name: tm[1], type: "Trait", startLine: ln, visibility: line.includes("pub") ? "public" : "private" }); }
	}
	return symbols;
}

function extractGoSymbols(content: string): SymbolDef[] {
	const symbols: SymbolDef[] = [];
	for (const [idx, line] of content.split("\n").entries()) {
		const ln = idx + 1;
		const fm = line.match(/^func\s+(?:\(\w+\s+\*?(\w+)\)\s+)?(\w+)/);
		if (fm) { symbols.push({ name: fm[2], type: fm[1] ? "Method" : "Function", startLine: ln, parentSymbol: fm[1] ?? undefined }); continue; }
		const tm = line.match(/^type\s+(\w+)\s+(struct|interface)/);
		if (tm) { symbols.push({ name: tm[1], type: tm[2] === "interface" ? "Interface" : "Struct", startLine: ln }); }
	}
	return symbols;
}

interface SymbolPatterns {
	classRe: RegExp;
	ifaceRe: RegExp;
	typeRe: RegExp;
	funcRe: RegExp;
	arrowRe: RegExp;
	methodRe: RegExp;
}

function extractSymbolsGeneric(content: string, p: SymbolPatterns): SymbolDef[] {
	const symbols: SymbolDef[] = [];
	let currentClass: string | undefined;

	for (const [idx, line] of content.split("\n").entries()) {
		const ln = idx + 1;
		const isExport = line.includes("export");

		const cm = line.match(p.classRe);
		if (cm) { currentClass = cm[1]; symbols.push({ name: cm[1], type: "Class", startLine: ln, visibility: isExport ? "public" : "private" }); continue; }

		const im = line.match(p.ifaceRe);
		if (im) { currentClass = undefined; symbols.push({ name: im[1], type: "Interface", startLine: ln, visibility: isExport ? "public" : "private" }); continue; }

		const tm = line.match(p.typeRe);
		if (tm) { symbols.push({ name: tm[1], type: "Type", startLine: ln, visibility: isExport ? "public" : "private" }); continue; }

		const fm = line.match(p.funcRe);
		if (fm) { symbols.push({ name: fm[1], type: currentClass ? "Method" : "Function", startLine: ln, parentSymbol: currentClass, visibility: isExport ? "public" : undefined, isAsync: line.includes("async") }); continue; }

		const am = line.match(p.arrowRe);
		if (am) { symbols.push({ name: am[1], type: currentClass ? "Method" : "Function", startLine: ln, parentSymbol: currentClass, visibility: isExport ? "public" : undefined, isAsync: line.includes("async") }); continue; }

		if (currentClass) {
			const mm = line.match(p.methodRe);
			if (mm && !line.match(/^(?:export|const|let|var|class|interface|type|function|if|for|while|switch|return)/)) {
				if (mm[3] === "constructor") continue;
				symbols.push({ name: mm[3], type: "Method", startLine: ln, parentSymbol: currentClass, visibility: mm[1] || "public", isAsync: line.includes("async"), isStatic: !!mm[2] });
			}
		}
	}
	return symbols;
}

function extractSymbols(content: string, language: string): SymbolDef[] {
	switch (language) {
		case "typescript": case "tsx": case "javascript": return extractTsSymbols(content);
		case "python": return extractPythonSymbols(content);
		case "rust": return extractRustSymbols(content);
		case "go": return extractGoSymbols(content);
		default: return [];
	}
}

// ── Containing symbol lookup ─────────────────────────────���──────────────────

function findContainingSymbol(callName: string, symbols: SymbolDef[], content: string): string | null {
	const lines = content.split("\n");
	for (let i = 0; i < lines.length; i++) {
		if (lines[i].includes(callName) && lines[i].includes("(")) {
			let best: SymbolDef | null = null;
			for (const sym of symbols) {
				if (sym.startLine <= i + 1 && (!best || sym.startLine > best.startLine)) best = sym;
			}
			if (best) return best.name;
		}
	}
	return null;
}

// ── Main extraction ─────────────────────────────────────────────────────────

export function extractGraph(content: string, filePath: string, language: string): ExtractedGraph {
	const nodes: GraphNode[] = [];
	const edges: EdgeSpec[] = [];
	const fName = fileName(filePath);

	nodes.push({ label: "File", properties: { name: fName, file_path: filePath, language } });

	const symbols = extractSymbols(content, language);
	for (const sym of symbols) {
		nodes.push({ label: sym.type, properties: { name: sym.name, file_path: filePath, start_line: sym.startLine, visibility: sym.visibility, is_async: sym.isAsync, is_static: sym.isStatic, language } });
		edges.push({ from: { file_path: filePath, symbol_name: fName }, to: { file_path: filePath, symbol_name: sym.name }, type: "CONTAINS" });
		if (sym.parentSymbol) {
			edges.push({ from: { file_path: filePath, symbol_name: sym.parentSymbol }, to: { file_path: filePath, symbol_name: sym.name }, type: "CONTAINS" });
		}
	}

	for (const imp of extractImports(content, filePath, language)) {
		edges.push({ from: { file_path: filePath, symbol_name: fName }, to: { file_path: imp.sourcePath, symbol_name: imp.symbolName }, type: "IMPORTS", properties: { alias: imp.alias, is_default: imp.isDefault } });
	}

	for (const h of extractHeritage(content, language)) {
		for (const parent of h.heritage) {
			edges.push({ from: { file_path: filePath, symbol_name: h.className }, to: { file_path: "*", symbol_name: parent.name }, type: parent.type === "extends" ? "EXTENDS" : "IMPLEMENTS" });
		}
	}

	for (const call of extractCalls(content)) {
		const caller = findContainingSymbol(call.functionName, symbols, content);
		if (caller) {
			edges.push({ from: { file_path: filePath, symbol_name: caller }, to: { file_path: "*", symbol_name: call.functionName }, type: "CALLS", properties: call.receiver ? { receiver: call.receiver } : undefined });
		}
	}

	return { nodes, edges };
}
