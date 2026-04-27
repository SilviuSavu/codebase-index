/**
 * keychain.ts — Read API keys from macOS Keychain as fallback
 *
 * Provides a fallback when environment variables aren't set.
 * Keys are stored via: security add-generic-password -a "openrouter" -s "OPENROUTER_API_KEY" -w <key>
 */

import { execSync } from "node:child_process";

function readKeychain(service: string, account: string): string | undefined {
  try {
    const result = execSync(
      `security find-generic-password -a "${account}" -s "${service}" -w ~/Library/Keychains/login.keychain-db 2>/dev/null`,
      { encoding: "utf-8", timeout: 5000 }
    );
    return result.trim() || undefined;
  } catch {
    return undefined;
  }
}

export function getOpenRouterKey(): string | undefined {
  return process.env.OPENROUTER_API_KEY || readKeychain("OPENROUTER_API_KEY", "openrouter");
}

export function getDeepInfraKey(): string | undefined {
  return process.env.DEEPINFRA_API_KEY || readKeychain("DEEPINFRA_API_KEY", "deepinfra");
}
