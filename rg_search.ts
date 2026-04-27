/**
 * rg-search.ts — Ripgrep-based file search
 *
 * Shells out to ripgrep for fast exact/regex matching against the workspace.
 * Returns results that can be fused with DB-based search via RRF.
 */

import { execFile } from "node:child_process";
import { join, relative } from "node:path";

export interface RgMatch {
	file_path: string;
	line_number: number;
	content: string;
}

const RG_PATH = join(process.env.HOME || "/root", ".pi/agent/bin/rg");

const LANG_TO_GLOB: Record<string, string> = {
	typescript: "*.{ts,tsx}",
	javascript: "*.{js,jsx,mjs,cjs}",
	python: "*.py",
	rust: "*.rs",
	go: "*.go",
	java: "*.java",
	kotlin: "*.{kt,kts}",
	swift: "*.swift",
	c: "*.{c,h}",
	cpp: "*.{cpp,cc,cxx,hpp}",
	csharp: "*.cs",
	ruby: "*.rb",
	php: "*.php",
	scala: "*.scala",
	lua: "*.lua",
	zig: "*.zig",
	dart: "*.dart",
	elixir: "*.{ex,exs}",
	haskell: "*.hs",
	bash: "*.{sh,bash,zsh}",
	css: "*.{css,scss,less}",
	html: "*.{html,htm}",
	vue: "*.vue",
	svelte: "*.svelte",
	sql: "*.sql",
};

const EXCLUDE_GLOBS = [
	"!{node_modules,.git,dist,build,target,coverage,vendor,.pi}",
	"!*.lock",
];

/**
 * Search workspace files using ripgrep.
 * Returns up to `limit` matches with file path, line number, and content.
 */
/**
 * Sanitize a query for use as an rg pattern.
 * - Strips newlines (rg patterns are single-line)
 * - Escapes regex metacharacters to treat the query as a literal string
 * - Truncates to a reasonable length
 */
function sanitizeRgQuery(query: string): string {
	let q = query.replace(/\n/g, " ").trim();
	// Escape regex metacharacters so the query is treated as literal text
	q = q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
	// Truncate to avoid absurdly long patterns
	if (q.length > 200) q = q.slice(0, 200);
	return q;
}

export async function rgSearch(
	rootDir: string,
	query: string,
	options?: {
		language?: string;
		limit?: number;
		signal?: AbortSignal;
	},
): Promise<RgMatch[]> {
	const limit = options?.limit ?? 30;
	const args = [
		"--json",
		"--max-count", String(limit),
		"--max-filesize", "1M",
		"--no-follow",
	];

	if (options?.language && LANG_TO_GLOB[options.language]) {
		args.push("--glob", LANG_TO_GLOB[options.language]);
	}

	for (const glob of EXCLUDE_GLOBS) {
		args.push("--glob", glob);
	}

	// Use -- to signal end of flags, preventing query tokens starting with `-`
	// from being interpreted as ripgrep flags.
	args.push("--", sanitizeRgQuery(query), rootDir);

	return new Promise<RgMatch[]>((resolve) => {
		const child = execFile(RG_PATH, args, {
			maxBuffer: 10 * 1024 * 1024,
			cwd: rootDir,
		}, (err, stdout) => {
			if (!stdout) {
				// rg exits with code 1 on no matches — not a real error
				const code = (err as NodeJS.ErrnoException | undefined)?.code;
				if (code !== 1 && err) {
					console.error("[codebase-index] rg error:", err.message);
				}
				resolve([]);
				return;
			}

			const matches: RgMatch[] = [];
			for (const line of stdout.split("\n")) {
				if (!line.trim()) continue;
				try {
					const parsed = JSON.parse(line);
					if (parsed.type === "match") {
						const data = parsed.data;
						const absPath: string = data.path?.text ?? data.path;
						matches.push({
							file_path: relative(rootDir, absPath),
							line_number: data.line_number,
							content: (data.lines?.text ?? data.lines ?? "").trimEnd(),
						});
						if (matches.length >= limit) break;
					}
				} catch {
					// Skip malformed lines
				}
			}
			resolve(matches);
		});

		// Wire up abort
		const signal = options?.signal;
		if (signal) {
			const onAbort = () => {
				child.kill("SIGTERM");
				resolve([]);
			};
			if (signal.aborted) {
				child.kill("SIGTERM");
				resolve([]);
			} else {
				signal.addEventListener("abort", onAbort, { once: true });
				child.on("close", () => signal.removeEventListener("abort", onAbort));
			}
		}
	});
}
