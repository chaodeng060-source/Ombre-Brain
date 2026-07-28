#!/usr/bin/env node

// Cross-platform launcher for the Python LMC-5 hook client.
// Claude Code ships with Node, while the Python executable is commonly named
// python3 on Linux and python (or py -3) on Windows.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const MAX_INPUT_BYTES = 512 * 1024;
const MAX_OUTPUT_BYTES = 4 * 1024 * 1024;
const input = readFileSync(0);

if (input.length > MAX_INPUT_BYTES) {
  writeFileSync(2, "ombre_lmc5_hook_error=launcher_input_too_large\n");
  process.exit(1);
}

let hookArgs = process.argv.slice(2);
const scriptName =
  hookArgs[0] === "session-breath" ? "session_breath.py" : "lmc5_hook.py";
if (hookArgs[0] === "session-breath") {
  hookArgs = [];
}
const script = join(dirname(fileURLToPath(import.meta.url)), scriptName);
const candidates =
  process.platform === "win32"
    ? [
        ["python", []],
        ["py", ["-3"]],
        ["python3", []],
      ]
    : [
        ["python3", []],
        ["python", []],
      ];

for (const [executable, prefixArgs] of candidates) {
  const result = spawnSync(
    executable,
    [...prefixArgs, script, ...hookArgs],
    {
      input,
      encoding: null,
      env: process.env,
      maxBuffer: MAX_OUTPUT_BYTES,
      windowsHide: true,
    },
  );

  if (result.error?.code === "ENOENT" && result.status === null) {
    continue;
  }
  if (result.stdout?.length) {
    writeFileSync(1, result.stdout);
  }
  if (result.stderr?.length) {
    writeFileSync(2, result.stderr);
  }
  if (result.error && result.status === null) {
    writeFileSync(2, "ombre_lmc5_hook_error=launcher_failed\n");
    process.exit(1);
  }
  process.exit(result.status ?? 1);
}

writeFileSync(2, "ombre_lmc5_hook_error=python_unavailable\n");
process.exit(1);
