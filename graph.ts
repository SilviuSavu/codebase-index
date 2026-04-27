/**
 * graph.ts — Apache AGE graph engine for codebase intelligence
 *
 * Manages a knowledge graph alongside the chunk index:
 * - Nodes: File, Module, Class, Function, Variable, Type, Enum
 * - Edges: CONTAINS, IMPORTS, CALLS, EXTENDS, IMPLEMENTS, REFERENCES
 *
 * Uses Apache AGE (Cypher on PostgreSQL) for graph storage and traversal.
 */

import pg, { type Pool, type PoolConfig } from "pg";

// ── Types ───────────────────────────────────────────────────────────────────

export type NodeType = "File" | "Module" | "Class" | "Interface" | "Function" | "Method" | "Variable" | "Type" | "Enum" | "Struct" | "Trait";

export type EdgeType = "CONTAINS" | "IMPORTS" | "CALLS" | "EXTENDS" | "IMPLEMENTS" | "REFERENCES" | "EXPORTS";

export interface GraphNode {
	id?: string;
	label: NodeType;
	properties: {
		name: string;
		file_path: string;
		start_line?: number;
		end_line?: number;
		chunk_id?: number;
		language?: string;
		visibility?: string;
		is_async?: boolean;
		is_static?: boolean;
	};
}

export interface GraphEdge {
	id?: string;
	label: EdgeType;
	properties: Record<string, unknown>;
}

export interface EdgeSpec {
	from: { file_path: string; symbol_name: string };
	to: { file_path: string; symbol_name: string };
	type: EdgeType;
	properties?: Record<string, unknown>;
}

export interface NeighborResult {
	node: GraphNode;
	edge: { type: EdgeType; direction: "incoming" | "outgoing" };
	pathLength: number;
}

// ── Connection management ───────────────────────────────────────────────────

const GRAPH_DB_CONFIG: PoolConfig = {
	host: "127.0.0.1",
	port: 5434,
	database: "codegraph",
	user: "codegraph",
	// password from docker-compose.yml, can be overridden via env  // NOSONAR: password is a default for local dev container
	password: process.env.CODEGRAPH_PASSWORD ?? "codegraph", // NOSONAR
	max: 5,
	idleTimeoutMillis: 30_000,
	connectionTimeoutMillis: 5_000,
};

const GRAPH_NAME = "codegraph";
let graphPool: Pool | null = null;

export function getGraphPool(config?: Partial<PoolConfig>): Pool {
	if (!graphPool) {
		graphPool = new pg.Pool({ ...GRAPH_DB_CONFIG, ...config });
		graphPool.on("error", (err: Error) => {
			console.error("[codebase-index/graph] pool error:", err.message);
		});
	}
	return graphPool;
}

export async function closeGraphPool(): Promise<void> {
	if (graphPool) {
		await graphPool.end();
		graphPool = null;
	}
}

export async function isGraphHealthy(pool: Pool): Promise<boolean> {
	try {
		await pool.query("SELECT 1");
		return true;
	} catch {
		return false;
	}
}

// ── agtype parsing ──────────────────────────────────────────────────────────

const AGTYPE_SUFFIX_RE = /::(vertex|edge|path|path_element)$/g;

/**
 * Parse Apache AGE agtype string into a JS object.
 * agtype looks like: {"id": 844424930131971, "label": "Function", "properties": {...}}::vertex
 */
function parseAgtype(val: string): Record<string, unknown> {
	if (!val) return {};
	const cleaned = val.replaceAll(AGTYPE_SUFFIX_RE, "");
	try {
		return JSON.parse(cleaned);
	} catch {
		return parseAgtypeWithUnquotedKeys(cleaned);
	}
}

function parseAgtypeWithUnquotedKeys(cleaned: string): Record<string, unknown> {
	try {
		// agtype might have unquoted keys
		const fixed = cleaned.replaceAll(/(\w+)\s*:/g, '"$1":');
		return JSON.parse(fixed);
	} catch {
		return {};
	}
}

function cypherStr(s: string): string {
	// NOSONAR: backslash literals cannot use String.raw — these ARE the escaped characters
	return s.replaceAll("\\", "\\\\").replaceAll("'", "\\'").replaceAll('"', '\\"');
}

function propsStr(props: Record<string, unknown>): string {
	const pairs = Object.entries(props)
		.filter(([, v]) => v !== undefined && v !== null)
		.map(([k, v]) => {
			if (typeof v === "string") return `${k}: '${cypherStr(v)}'`;
			if (typeof v === "boolean") return `${k}: ${v}`;
			if (typeof v === "number") return `${k}: ${v}`;
			return `${k}: '${cypherStr(JSON.stringify(v))}'`;
		});
	return `{${pairs.join(", ")}}`;
}

// ── Schema management ───────────────────────────────────────────────────────

export async function ensureGraphSchema(pool: Pool): Promise<void> {
	await pool.query(`LOAD 'age'`);
	await pool.query(`SET search_path = ag_catalog, "$user", public`);

	try {
		await pool.query(`SELECT create_graph('${GRAPH_NAME}')`);
	} catch (err) {
		const msg = (err as Error).message;
		if (!msg.includes("already exists")) {
			console.error("[codebase-index/graph] create_graph:", msg);
		}
	}
}

async function cypherQuery<T extends Record<string, unknown>>(
	pool: Pool,
	query: string,
	returnVars: string[],
): Promise<T[]> {
	await pool.query(`SET search_path = ag_catalog, "$user", public`);
	const aliases = returnVars.map((v) => `${v} agtype`).join(", ");
	const sql = `SELECT * FROM cypher('${GRAPH_NAME}', $$${query}$$) AS (${aliases});`;
	const { rows } = await pool.query(sql);
	return rows.map((row) => {
		const parsed: Record<string, unknown> = {};
		for (const v of returnVars) {
			if (row[v]) parsed[v] = parseAgtype(String(row[v]));
		}
		return parsed as T;
	});
}

// ── Node operations ─────────────────────────────────────────────────────────

interface NodeLookup {
	vertexId: string;
	properties: Record<string, unknown>;
}

export async function upsertNodes(
	pool: Pool,
	nodes: GraphNode[],
): Promise<Map<string, NodeLookup>> {
	if (nodes.length === 0) return new Map<string, NodeLookup>();

	const lookup = new Map<string, NodeLookup>();
	const BATCH = 50;
	for (let i = 0; i < nodes.length; i += BATCH) {
		const batch = nodes.slice(i, i + BATCH);
		await upsertNodeBatch(pool, batch, lookup);
	}

	return lookup;
}

async function upsertNodeBatch(
	pool: Pool,
	batch: GraphNode[],
	lookup: Map<string, NodeLookup>,
): Promise<void> {
	const creates = batch.map((node, idx) => {
		const varName = `n${idx}`;
		const props = propsStr({
			name: node.properties.name,
			file_path: node.properties.file_path,
			start_line: node.properties.start_line,
			end_line: node.properties.end_line,
			chunk_id: node.properties.chunk_id,
			language: node.properties.language,
			visibility: node.properties.visibility,
			is_async: node.properties.is_async,
			is_static: node.properties.is_static,
		});
		const escapedPath = cypherStr(node.properties.file_path);
		const escapedName = cypherStr(node.properties.name);
		return [
			`MERGE (${varName}:${node.label} {file_path: '${escapedPath}', name: '${escapedName}'})`,
			`SET ${varName} += ${props}`,
		].join("\n");
	}).join("\n");

	const returns = batch.map((_, idx) => `n${idx}`).join(", ");
	const returnAliases = batch.map((_, idx) => `n${idx} agtype`).join(", ");

	try {
		await pool.query(`SET search_path = ag_catalog, "$user", public`);
		const cypherBody = `${creates}\n  RETURN ${returns}`;
		const sql = `SELECT * FROM cypher('${GRAPH_NAME}', $$${cypherBody}$$) AS (${returnAliases});`;
		const { rows } = await pool.query(sql);

		if (rows.length > 0) {
			const row = rows[0];
			for (let j = 0; j < batch.length; j++) {
				const node = batch[j];
				const key = `${node.properties.file_path}::${node.properties.name}::${node.label}`;
				const raw = row[`n${j}`];
				if (raw) {
					const parsed = parseAgtype(String(raw));
					const id = typeof parsed.id === "number" ? String(parsed.id) : "";
					lookup.set(key, { vertexId: id, properties: (parsed.properties ?? {}) as Record<string, unknown> });
				}
			}
		}
	} catch (err) {
		console.error("[codebase-index/graph] node upsert error:", (err as Error).message);
	}
}

// ── Edge operations ─────────────────────────────────────────────────────────

export async function upsertEdges(pool: Pool, edges: EdgeSpec[]): Promise<number> {
	if (edges.length === 0) return 0;

	let created = 0;
	for (const e of edges) {
		try {
			await upsertSingleEdge(pool, e);
			created++;
		} catch (err) {
			// Only log non-wildcard edge failures (wildcard edges are expected to fail)
			if (e.to.file_path !== "*") {
				console.error("[codebase-index/graph] edge error:", (err as Error).message, `(${e.type}: ${e.from.symbol_name} -> ${e.to.symbol_name})`);
			}
		}
	}
	return created;
}

async function upsertSingleEdge(pool: Pool, e: EdgeSpec): Promise<void> {
	const extraProps = e.properties ? ` += ${propsStr(e.properties)}` : "";
	const escapedFromPath = cypherStr(e.from.file_path);
	const escapedFromName = cypherStr(e.from.symbol_name);
	const escapedToPath = cypherStr(e.to.file_path);
	const escapedToName = cypherStr(e.to.symbol_name);

	const mergeLine = e.properties
		? `MERGE (src)-[r:${e.type}]->(tgt) SET r${extraProps}`
		: `MERGE (src)-[r:${e.type}]->(tgt)`;

	const cypher = [
		`MATCH (src {file_path: '${escapedFromPath}', name: '${escapedFromName}'})`,
		`MATCH (tgt {file_path: '${escapedToPath}', name: '${escapedToName}'})`,
		mergeLine,
		"RETURN r",
	].join("\n");

	await pool.query(`SET search_path = ag_catalog, "$user", public`);
	const sql = `SELECT * FROM cypher('${GRAPH_NAME}', $$${cypher}$$) AS (r agtype);`;
	await pool.query(sql);
}

// ── Helper: map raw graph result to NeighborResult ──────────────────────────

function getStringProp(props: Record<string, unknown>, key: string): string {
	const val = props[key];
	return typeof val === "string" ? val : "";
}

function toNeighborResult(
	props: Record<string, unknown>,
	label: string,
	edgeType: string,
	direction: "incoming" | "outgoing",
	pathLen: number,
): NeighborResult {
	return {
		node: {
			label: label as NodeType,
			properties: {
				name: getStringProp(props, "name"),
				file_path: getStringProp(props, "file_path"),
				start_line: typeof props.start_line === "number" ? props.start_line : undefined,
				end_line: typeof props.end_line === "number" ? props.end_line : undefined,
				chunk_id: typeof props.chunk_id === "number" ? props.chunk_id : undefined,
				language: typeof props.language === "string" ? props.language : undefined,
			},
		},
		edge: { type: edgeType as EdgeType, direction },
		pathLength: pathLen,
	};
}

// ── Graph queries ───────────────────────────────────────────────────────────

export async function getNeighborhood(
	pool: Pool,
	symbolName: string,
	filePath: string,
	options?: { maxDepth?: number; maxResults?: number; direction?: "both" | "incoming" | "outgoing" },
): Promise<NeighborResult[]> {
	const maxDepth = options?.maxDepth ?? 2;
	const maxResults = options?.maxResults ?? 20;

	const cypher = [
		`MATCH (start {file_path: '${cypherStr(filePath)}', name: '${cypherStr(symbolName)}'})`,
		"MATCH path = (start)-[r]-(other)",
		`WHERE length(path) <= ${maxDepth}`,
		"WITH other, r, length(path) AS pathLen, startNode(r) AS edgeSrc",
		"RETURN other, type(r) AS edge_type,",
		"       CASE WHEN edgeSrc = start THEN 'outgoing' ELSE 'incoming' END AS direction,",
		"       pathLen",
		"ORDER BY pathLen ASC",
		`LIMIT ${maxResults}`,
	].join("\n");

	const results = await cypherQuery<{
		other: { label?: string; properties?: Record<string, unknown> };
		edge_type: string;
		direction: string;
		pathLen: number;
	}>(pool, cypher, ["other", "edge_type", "direction", "pathLen"]);

	return results.map((r) => {
		const props = r.other?.properties ?? {};
		const dir = r.direction === "outgoing" ? "outgoing" as const : "incoming" as const;
		return toNeighborResult(props, r.other?.label ?? "Unknown", String(r.edge_type), dir, Number(r.pathLen));
	});
}

export async function getCallers(
	pool: Pool,
	symbolName: string,
	filePath: string,
	maxDepth: number = 3,
): Promise<NeighborResult[]> {
	return getNeighborhood(pool, symbolName, filePath, { maxDepth, direction: "incoming" });
}

export async function getCallees(
	pool: Pool,
	symbolName: string,
	filePath: string,
	maxDepth: number = 3,
): Promise<NeighborResult[]> {
	return getNeighborhood(pool, symbolName, filePath, { maxDepth, direction: "outgoing" });
}

export async function getInheritance(
	pool: Pool,
	className: string,
	filePath: string,
): Promise<NeighborResult[]> {
	const cypher = [
		`MATCH (start {file_path: '${cypherStr(filePath)}', name: '${cypherStr(className)}'})`,
		"MATCH path = (start)-[:EXTENDS|IMPLEMENTS*1..5]-(related)",
		"WITH related, relationships(path) AS rels, length(path) AS pathLen",
		"RETURN related, type(last(rels)) AS edge_type, pathLen",
		"ORDER BY pathLen ASC",
		"LIMIT 20",
	].join("\n");

	const results = await cypherQuery<{
		related: { label?: string; properties?: Record<string, unknown> };
		edge_type: string;
		pathLen: number;
	}>(pool, cypher, ["related", "edge_type", "pathLen"]);

	return results.map((r) => {
		const props = r.related?.properties ?? {};
		return toNeighborResult(props, r.related?.label ?? "Unknown", String(r.edge_type), "outgoing" as const, Number(r.pathLen));
	});
}

export async function getFileContents(
	pool: Pool,
	filePath: string,
): Promise<NeighborResult[]> {
	return getNeighborhood(pool, "*", filePath, { maxDepth: 1, direction: "outgoing", maxResults: 50 });
}

export async function getImpactRadius(
	pool: Pool,
	symbolName: string,
	filePath: string,
	maxDepth: number = 4,
): Promise<NeighborResult[]> {
	const maxHops = maxDepth;
	const cypher = [
		`MATCH (start {file_path: '${cypherStr(filePath)}', name: '${cypherStr(symbolName)}'})`,
		`MATCH path = (start)<-[:CALLS|REFERENCES|EXTENDS|IMPLEMENTS*1..${maxHops}]-(dependent)`,
		"WITH dependent, relationships(path) AS rels, length(path) AS pathLen",
		"RETURN dependent, type(last(rels)) AS edge_type, pathLen",
		"ORDER BY pathLen ASC",
		"LIMIT 30",
	].join("\n");

	const results = await cypherQuery<{
		dependent: { label?: string; properties?: Record<string, unknown> };
		edge_type: string;
		pathLen: number;
	}>(pool, cypher, ["dependent", "edge_type", "pathLen"]);

	return results.map((r) => {
		const props = r.dependent?.properties ?? {};
		return toNeighborResult(props, r.dependent?.label ?? "Unknown", String(r.edge_type), "incoming" as const, Number(r.pathLen));
	});
}

export async function getDependents(
	pool: Pool,
	filePath: string,
): Promise<NeighborResult[]> {
	const cypher = [
		`MATCH (f:File {file_path: '${cypherStr(filePath)}'})<-[:IMPORTS*1..3]-(dependent)`,
		"RETURN DISTINCT dependent, 'IMPORTS' AS edge_type, 1 AS pathLen",
		"LIMIT 20",
	].join("\n");

	const results = await cypherQuery<{
		dependent: { label?: string; properties?: Record<string, unknown> };
		edge_type: string;
		pathLen: number;
	}>(pool, cypher, ["dependent", "edge_type", "pathLen"]);

	return results.map((r) => {
		const props = r.dependent?.properties ?? {};
		return toNeighborResult(props, r.dependent?.label ?? "Unknown", "IMPORTS", "incoming" as const, Number(r.pathLen));
	});
}

// ── Graph statistics ───────────────────────────────────────────────────────

async function fetchNodeCount(pool: Pool): Promise<number> {
	const nodeCount = await cypherQuery<{ total: Record<string, unknown> }>(
		pool, "MATCH (n) RETURN count(n) AS total", ["total"],
	);
	return Number(nodeCount[0]?.total?.["count(n)"] ?? nodeCount.length);
}

async function fetchEdgeCount(pool: Pool): Promise<number> {
	const edgeCount = await cypherQuery<{ total: Record<string, unknown> }>(
		pool, "MATCH ()-[r]->() RETURN count(r) AS total", ["total"],
	);
	return Number(edgeCount[0]?.total?.["count(r)"] ?? edgeCount.length);
}

async function fetchNodeTypeBreakdown(pool: Pool): Promise<Record<string, number>> {
	const nodeTypes: Record<string, number> = {};
	try {
		const rows = await cypherQuery<{ result: Record<string, unknown> }>(
			pool, "MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt", ["result"],
		);
		for (const r of rows) {
			const labels = r.result?.["labels(n)"];
			const cnt = Number(r.result?.["count(n)"] ?? 0);
			if (Array.isArray(labels)) {
				for (const l of labels) nodeTypes[String(l)] = (nodeTypes[String(l)] ?? 0) + cnt;
			}
		}
	} catch { /* labels() not available in all AGE versions */ }
	return nodeTypes;
}

async function fetchEdgeTypeBreakdown(pool: Pool): Promise<Record<string, number>> {
	const edgeTypes: Record<string, number> = {};
	try {
		const rows = await cypherQuery<{ result: Record<string, unknown> }>(
			pool, "MATCH ()-[r]->() RETURN type(r) AS edge_type, count(r) AS cnt", ["result"],
		);
		for (const r of rows) {
			const raw = r.result?.["type(r)"];
			const t = typeof raw === "string" ? raw : "unknown";
			edgeTypes[t] = Number(r.result?.["count(r)"] ?? 0);
		}
	} catch { /* type() not available in all AGE versions */ }
	return edgeTypes;
}

export async function getGraphStats(pool: Pool): Promise<{
	nodeCount: number;
	edgeCount: number;
	nodeTypes: Record<string, number>;
	edgeTypes: Record<string, number>;
}> {
	try {
		const [nodeCount, edgeCount, nodeTypes, edgeTypes] = await Promise.all([
			fetchNodeCount(pool),
			fetchEdgeCount(pool),
			fetchNodeTypeBreakdown(pool),
			fetchEdgeTypeBreakdown(pool),
		]);
		return { nodeCount, edgeCount, nodeTypes, edgeTypes };
	} catch (err) {
		console.error("[codebase-index/graph] stats error:", (err as Error).message);
		return { nodeCount: 0, edgeCount: 0, nodeTypes: {}, edgeTypes: {} };
	}
}

// ── Graph mutation ──────────────────────────────────────────────────────────

export async function clearProjectGraph(pool: Pool, projectPrefix: string): Promise<void> {
	const cypher = `MATCH (n) WHERE n.file_path STARTS WITH '${cypherStr(projectPrefix)}' DETACH DELETE n`;
	try {
		await cypherQuery(pool, cypher, []);
	} catch (err) {
		console.error("[codebase-index/graph] clear error:", (err as Error).message);
	}
}

export async function deleteFileFromGraph(pool: Pool, filePath: string): Promise<void> {
	const cypher = `MATCH (n {file_path: '${cypherStr(filePath)}'}) DETACH DELETE n`;
	try {
		await cypherQuery(pool, cypher, []);
	} catch (err) {
		console.error("[codebase-index/graph] delete file error:", (err as Error).message);
	}
}

// ── Context expansion ───────────────────────────────────────────────────────

interface ContextChunk {
	file_path: string;
	symbol_name: string | null;
	symbol_type: string | null;
}

interface ExpandedSymbol {
	file_path: string;
	symbol_name: string;
	symbol_type: string;
	relation: string;
}

export async function expandContext(
	pool: Pool,
	chunks: ContextChunk[],
	maxExtra: number = 10,
): Promise<ExpandedSymbol[]> {
	const extra: ExpandedSymbol[] = [];
	const seen = new Set<string>();

	for (const c of chunks) {
		if (c.symbol_name) seen.add(`${c.file_path}::${c.symbol_name}`);
	}

	for (const chunk of chunks) {
		if (!chunk.symbol_name || extra.length >= maxExtra) break;
		await expandChunkNeighbors(pool, chunk.symbol_name, chunk.file_path, extra, seen, maxExtra);
	}

	return extra;
}

async function expandChunkNeighbors(
	pool: Pool,
	symbolName: string,
	filePath: string,
	extra: ExpandedSymbol[],
	seen: Set<string>,
	maxExtra: number,
): Promise<void> {
	try {
		const neighbors = await getNeighborhood(pool, symbolName, filePath, { maxDepth: 2, maxResults: 5 });
		for (const n of neighbors) {
			if (extra.length >= maxExtra) break;
			const name = n.node.properties.name;
			const path = n.node.properties.file_path;
			if (!name || !path) continue;
			const key = `${path}::${name}`;
			if (seen.has(key)) continue;
			seen.add(key);
			extra.push({ file_path: path, symbol_name: name, symbol_type: n.node.label, relation: `${n.edge.direction}:${n.edge.type}` });
		}
	} catch { /* skip expansion failures */ }
}
