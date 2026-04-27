/**
 * db.ts — PostgreSQL connection pool + schema management
 *
 * Manages the PG connection pool, ensures extensions are installed,
 * and creates/migrates the `code_chunks` table with all required indexes.
 */

import type { Pool, PoolConfig } from "pg";
import pg from "pg";

const SCHEMA_SQL = `
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

CREATE TABLE IF NOT EXISTS code_chunks (
    id          BIGSERIAL PRIMARY KEY,
    project     TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    language    TEXT,
    symbol_type TEXT,
    symbol_name TEXT,
    start_line  INT,
    end_line    INT,
    content     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding   vector(4096),
    metadata    JSONB DEFAULT '{}',
    indexed_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_project ON code_chunks (project);
CREATE INDEX IF NOT EXISTS idx_trgm ON code_chunks USING gin (content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_fts ON code_chunks USING gin (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_hash ON code_chunks (content_hash);
CREATE INDEX IF NOT EXISTS idx_meta ON code_chunks USING gin (metadata);
`;

export interface ChunkRow {
	id?: number;
	project: string;
	file_path: string;
	language: string | null;
	symbol_type: string | null;
	symbol_name: string | null;
	start_line: number | null;
	end_line: number | null;
	content: string;
	content_hash: string;
	embedding?: number[] | null;
	metadata?: Record<string, unknown>;
	indexed_at?: Date;
}

export interface SearchResult {
	id: number;
	file_path: string;
	language: string | null;
	symbol_type: string | null;
	symbol_name: string | null;
	start_line: number | null;
	end_line: number | null;
	content: string;
	score: number;
}

const DEFAULT_CONFIG: PoolConfig = {
	host: "127.0.0.1",
	port: 5433,
	database: "codebase",
	user: "codebase",
	password: "codebase",
	max: 10,
	idleTimeoutMillis: 30_000,
	connectionTimeoutMillis: 5_000,
};

let pool: Pool | null = null;

export function getPool(config?: Partial<PoolConfig>): Pool {
	if (!pool) {
		pool = new pg.Pool({ ...DEFAULT_CONFIG, ...config });
		pool.on("error", (err: Error) => {
			console.error("[codebase-index] unexpected pool error:", err.message);
		});
	}
	return pool;
}

export async function closePool(): Promise<void> {
	if (pool) {
		await pool.end();
		pool = null;
	}
}

export async function ensureSchema(pool: Pool): Promise<void> {
	await pool.query(SCHEMA_SQL);
	const { rows } = await pool.query(
		"SELECT count(*) as cnt FROM code_chunks WHERE embedding IS NOT NULL",
	);
	const count = Number(rows[0]?.cnt ?? 0);
	if (count > 100) {
		await pool.query(`
      CREATE INDEX IF NOT EXISTS idx_vec ON code_chunks
      USING hnsw (embedding vector_cosine_ops)
      WITH (m = 16, ef_construction = 64)
    `);
	}
}

interface ExistingRow {
	id: number;
	content_hash: string;
}

/**
 * Sync chunks for a single file: delete stale, insert new.
 */
async function syncFileChunks(
	client: pg.PoolClient,
	newChunks: ChunkRow[],
	existing: ExistingRow[],
): Promise<number> {
	let inserted = 0;
	const existingByHash = new Map(
		existing.map((r) => [r.content_hash, r.id] as const),
	);
	const newHashes = new Set(newChunks.map((c) => c.content_hash));

	for (const [hash, id] of existingByHash) {
		if (!newHashes.has(hash)) {
			await client.query("DELETE FROM code_chunks WHERE id = $1", [id]);
		}
	}

	for (const chunk of newChunks) {
		if (existingByHash.has(chunk.content_hash)) continue;
		await client.query(
			`INSERT INTO code_chunks (project, file_path, language, symbol_type, symbol_name, start_line, end_line, content, content_hash, metadata)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)`,
			[
				chunk.project,
				chunk.file_path,
				chunk.language,
				chunk.symbol_type,
				chunk.symbol_name,
				chunk.start_line,
				chunk.end_line,
				chunk.content,
				chunk.content_hash,
				JSON.stringify(chunk.metadata ?? {}),
			],
		);
		inserted++;
	}
	return inserted;
}

/** Upsert a batch of chunks. Deletes stale chunks for the same project+file. */
export async function upsertChunks(
	pool: Pool,
	chunks: ChunkRow[],
): Promise<number> {
	if (chunks.length === 0) return 0;

	let totalInserted = 0;
	const client = await pool.connect();
	try {
		await client.query("BEGIN");

		const byFile = new Map<string, ChunkRow[]>();
		for (const chunk of chunks) {
			const key = `${chunk.project}\0${chunk.file_path}`;
			let list = byFile.get(key);
			if (!list) {
				list = [];
				byFile.set(key, list);
			}
			list.push(chunk);
		}

		for (const [key, fileChunks] of byFile) {
			const sepIdx = key.indexOf("\0");
			const project = key.slice(0, sepIdx);
			const filePath = key.slice(sepIdx + 1);

			const { rows: existing } = await client.query<ExistingRow>(
				"SELECT id, content_hash FROM code_chunks WHERE project = $1 AND file_path = $2",
				[project, filePath],
			);

			totalInserted += await syncFileChunks(client, fileChunks, existing);
		}

		await client.query("COMMIT");
		return totalInserted;
	} catch (err) {
		await client.query("ROLLBACK");
		throw err;
	} finally {
		client.release();
	}
}

/** Update embeddings for chunks */
export async function updateEmbeddings(
	pool: Pool,
	embeddings: Map<number, number[]>,
): Promise<void> {
	if (embeddings.size === 0) return;
	const client = await pool.connect();
	try {
		await client.query("BEGIN");
		for (const [id, emb] of embeddings) {
			await client.query(
				"UPDATE code_chunks SET embedding = $1 WHERE id = $2",
				[`[${emb.join(",")}]`, id],
			);
		}
		await client.query("COMMIT");
	} catch (err) {
		await client.query("ROLLBACK");
		throw err;
	} finally {
		client.release();
	}
}

/** Get chunks without embeddings, limited to batch size */
export async function getChunksWithoutEmbeddings(
	pool: Pool,
	project: string,
	limit: number,
): Promise<ChunkRow[]> {
	const { rows } = await pool.query(
		"SELECT id, content FROM code_chunks WHERE project = $1 AND embedding IS NULL ORDER BY id LIMIT $2",
		[project, limit],
	);
	return rows as ChunkRow[];
}

/** Delete all chunks for a project+file */
export async function deleteFileChunks(
	pool: Pool,
	project: string,
	filePath: string,
): Promise<void> {
	await pool.query(
		"DELETE FROM code_chunks WHERE project = $1 AND file_path = $2",
		[project, filePath],
	);
}

/** Get all distinct file paths for a project */
export async function getIndexedFiles(
	pool: Pool,
	project: string,
): Promise<string[]> {
	const { rows } = await pool.query(
		"SELECT DISTINCT file_path FROM code_chunks WHERE project = $1",
		[project],
	);
	return rows.map((r: { file_path: string }) => r.file_path);
}

/** Get count of indexed chunks for a project */
export async function getChunkCount(
	pool: Pool,
	project: string,
): Promise<number> {
	const { rows } = await pool.query(
		"SELECT count(*) as cnt FROM code_chunks WHERE project = $1",
		[project],
	);
	return Number(rows[0]?.cnt ?? 0);
}

/** Check if database is reachable */
export async function isHealthy(pool: Pool): Promise<boolean> {
	try {
		await pool.query("SELECT 1");
		return true;
	} catch {
		return false;
	}
}
